"""Train YOLOv8x with scene-grouped Origin-anchored consistency.

The wrapper adds synchronized sampling and training-only feature hooks while
preserving the standard serialized inference graph.
"""

from __future__ import print_function

import argparse
import csv
import math
import os
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torchvision.ops import roi_align

try:
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOCAL_RANK, RANK
    from ultralytics.utils.torch_utils import (
        torch_distributed_zero_first,
        unwrap_model,
    )
except Exception:
    DetectionTrainer = object


def read_scene_manifest(path):
    """Index manifest rows by displayed, resolved, and basename image paths."""

    rows = []
    lookup = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            image_path = Path(row["yolo_image_path"])
            for key in (str(image_path), str(image_path.resolve()), image_path.name):
                lookup[key] = row
    return rows, lookup


def manifest_row_for_path(path, lookup):
    image_path = Path(str(path))
    for key in (str(image_path), str(image_path.resolve()), image_path.name):
        if key in lookup:
            return lookup[key]
    return None


class OriginGroupedDistributedSampler(Sampler):
    """Yield local batches as repeated Origin-plus-styled scene groups."""

    def __init__(
        self,
        dataset,
        manifest_lookup,
        batch_size,
        rank,
        world_size,
        styles_per_anchor,
        seed,
    ):
        self.batch_size = int(batch_size)
        self.rank = max(int(rank), 0)
        self.world_size = max(int(world_size), 1)
        self.styles_per_anchor = int(styles_per_anchor)
        self.group_size = 1 + self.styles_per_anchor
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size % self.group_size:
            raise ValueError(
                "Per-rank batch size {} must be divisible by scene group size {}.".format(
                    self.batch_size, self.group_size
                )
            )
        self.groups_per_batch = self.batch_size // self.group_size

        grouped = {}
        missing = []
        for index, image_path in enumerate(dataset.im_files):
            row = manifest_row_for_path(image_path, manifest_lookup)
            if row is None:
                missing.append(str(image_path))
                continue
            scene = row["scene"]
            style = row.get("source_style") or row.get("style", "").split(":")[-1]
            entry = grouped.setdefault(scene, {"origin": None, "styled": []})
            if style.casefold() == "origin":
                entry["origin"] = index
            else:
                entry["styled"].append(index)
        if missing:
            raise RuntimeError(
                "{} YOLO images are absent from the scene manifest; first: {}".format(
                    len(missing), missing[0]
                )
            )
        self.groups = {
            scene: entry
            for scene, entry in grouped.items()
            if entry["origin"] is not None and entry["styled"]
        }
        if not self.groups:
            raise RuntimeError("No Origin-plus-style scene groups were found.")
        self.scenes = sorted(self.groups)
        scene_multiple = self.world_size * self.groups_per_batch
        self.total_scene_slots = int(
            math.ceil(len(self.scenes) / float(scene_multiple)) * scene_multiple
        )
        self.local_scene_slots = self.total_scene_slots // self.world_size

    def __len__(self):
        return self.local_scene_slots * self.group_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1000003)
        scenes = list(self.scenes)
        rng.shuffle(scenes)
        if len(scenes) < self.total_scene_slots:
            scenes.extend(scenes[: self.total_scene_slots - len(scenes)])
        rank_scenes = scenes[self.rank : self.total_scene_slots : self.world_size]
        indices = []
        for scene in rank_scenes:
            entry = self.groups[scene]
            styled = list(entry["styled"])
            if len(styled) >= self.styles_per_anchor:
                chosen = rng.sample(styled, self.styles_per_anchor)
            else:
                chosen = [
                    styled[(self.epoch + offset) % len(styled)]
                    for offset in range(self.styles_per_anchor)
                ]
            indices.append(entry["origin"])
            indices.extend(chosen)
        return iter(indices)


def feature_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [value[key] for key in sorted(value) if isinstance(value[key], torch.Tensor)]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def global_embeddings(features):
    pooled = [
        value.float().mean(dim=(-2, -1))
        for value in features
        if value.ndim == 4
    ]
    if not pooled:
        return None
    return F.normalize(torch.cat(pooled, dim=1), dim=1)


def xywh_to_feature_rois(pairs, feature, side):
    height, width = feature.shape[-2:]
    rows = []
    for pair in pairs:
        sample_index = pair["student_index"] if side == "student" else pair["teacher_index"]
        box = pair["student_box"] if side == "student" else pair["teacher_box"]
        x, y, w, h = [float(value) for value in box]
        x1 = max(0.0, min(float(width), (x - w / 2.0) * width))
        y1 = max(0.0, min(float(height), (y - h / 2.0) * height))
        x2 = max(x1 + 1.0e-3, min(float(width), (x + w / 2.0) * width))
        y2 = max(y1 + 1.0e-3, min(float(height), (y + h / 2.0) * height))
        rows.append([float(sample_index), x1, y1, x2, y2])
    return torch.tensor(rows, device=feature.device, dtype=torch.float32)


def roi_embeddings(features, pairs, side):
    pooled = []
    for value in features:
        if value.ndim != 4:
            continue
        rois = xywh_to_feature_rois(pairs, value, side)
        aligned = roi_align(
            value.float(),
            rois,
            output_size=(1, 1),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        pooled.append(aligned.flatten(1))
    if not pooled:
        return None
    return F.normalize(torch.cat(pooled, dim=1), dim=1)


def sorted_objects(batch, sample_index):
    mask = batch["batch_idx"].long() == int(sample_index)
    boxes = batch["bboxes"][mask]
    classes = batch["cls"][mask].flatten()
    if boxes.numel() == 0:
        return boxes, classes
    detached = torch.cat([classes[:, None], boxes], dim=1).detach().cpu().tolist()
    order = sorted(range(len(detached)), key=lambda idx: tuple(detached[idx]))
    order_tensor = torch.tensor(order, device=boxes.device, dtype=torch.long)
    return boxes[order_tensor], classes[order_tensor]


def build_object_pairs(batch, scene_ids, origin_indices, styles_per_scene, max_objects):
    """Pair GT objects by scene, canonical ordering, and matching class."""

    origin_by_scene = {
        int(scene_ids[index]): int(index)
        for index in origin_indices.detach().cpu().tolist()
    }
    teacher_index_by_origin = {
        int(origin_index): local_index
        for local_index, origin_index in enumerate(origin_indices.detach().cpu().tolist())
    }
    pairs = []
    for student_index in styles_per_scene.detach().cpu().tolist():
        scene_id = int(scene_ids[student_index])
        origin_index = origin_by_scene.get(scene_id)
        if origin_index is None:
            continue
        student_boxes, student_classes = sorted_objects(batch, student_index)
        teacher_boxes, teacher_classes = sorted_objects(batch, origin_index)
        count = min(
            int(student_boxes.shape[0]),
            int(teacher_boxes.shape[0]),
            int(max_objects),
        )
        for object_index in range(count):
            if int(student_classes[object_index]) != int(teacher_classes[object_index]):
                continue
            pairs.append(
                {
                    "student_index": int(student_index),
                    "teacher_index": teacher_index_by_origin[origin_index],
                    "student_box": student_boxes[object_index].detach().cpu().tolist(),
                    "teacher_box": teacher_boxes[object_index].detach().cpu().tolist(),
                }
            )
    return pairs


def ramp_weight(step, start, end):
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    return float(step - start) / float(max(1, end - start))


def find_detect_module(model):
    inner = unwrap_model(model)
    layers = getattr(inner, "model", None)
    if layers is None:
        raise RuntimeError("Unable to find YOLO model layers.")
    for module in reversed(list(layers)):
        if module.__class__.__name__.lower().endswith("detect"):
            return module
    return list(layers)[-1]


def strip_oamsc_state(model):
    inner = unwrap_model(model)
    loss = getattr(inner, "loss", None)
    if getattr(loss, "__name__", "") == "loss_with_oamsc":
        try:
            delattr(inner, "loss")
        except AttributeError:
            pass


class OAMSCDetectionTrainer(DetectionTrainer):
    """Add OAMSC grouping, geometry, hooks, and losses to DetectionTrainer.

    The patch is training-local; serialization removes hook and auxiliary-loss
    state so retained YOLO checkpoints use the standard inference graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = os.environ.get("YOLO_OAMSC_SCENE_MANIFEST")
        if not manifest:
            raise RuntimeError("YOLO_OAMSC_SCENE_MANIFEST is required.")
        _rows, self.scene_lookup = read_scene_manifest(manifest)
        self.global_weight = float(os.environ.get("YOLO_OAMSC_GLOBAL_WEIGHT", "0.10"))
        self.roi_weight = float(os.environ.get("YOLO_OAMSC_ROI_WEIGHT", "0.05"))
        self.ramp_start = int(os.environ.get("YOLO_OAMSC_RAMP_START", "2000"))
        self.ramp_end = int(os.environ.get("YOLO_OAMSC_RAMP_END", "6000"))
        self.styles_per_anchor = int(os.environ.get("YOLO_OAMSC_STYLES_PER_ANCHOR", "3"))
        self.max_objects = int(os.environ.get("YOLO_OAMSC_MAX_OBJECTS", "64"))
        self.oamsc_seed = int(os.environ.get("YOLO_OAMSC_SEED", "20260723"))
        self.hflip_prob = float(os.environ.get("YOLO_OAMSC_HFLIP_PROB", "0.5"))
        self._student_features = {}
        self._teacher_features = {}
        self._student_hook = None
        self._teacher_hook = None
        self._step_epoch = None
        self._batch_in_epoch = 0
        self._current_step = 0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self._patch_loss(model)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = OriginGroupedDistributedSampler(
            dataset=dataset,
            manifest_lookup=self.scene_lookup,
            batch_size=batch_size,
            rank=rank,
            world_size=self.world_size,
            styles_per_anchor=self.styles_per_anchor,
            seed=self.oamsc_seed,
        )
        batches = len(sampler) // batch_size
        workers = min(
            (os.cpu_count() or 1) // max(torch.cuda.device_count(), 1),
            self.args.workers,
            0 if batches <= 1 else batches,
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + max(rank, 0))
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            sampler=sampler,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        files = batch.get("im_file") or batch.get("im_files") or []
        scene_to_id = {}
        scene_ids = []
        origin_flags = []
        for value in files:
            row = manifest_row_for_path(value, self.scene_lookup)
            if row is None:
                raise RuntimeError("Training image is absent from scene manifest: {}".format(value))
            scene = row["scene"]
            if scene not in scene_to_id:
                scene_to_id[scene] = len(scene_to_id)
            scene_ids.append(scene_to_id[scene])
            style = row.get("source_style") or row.get("style", "").split(":")[-1]
            origin_flags.append(style.casefold() == "origin")
        batch["scene_ids"] = torch.tensor(scene_ids, device=self.device, dtype=torch.long)
        batch["is_origin"] = torch.tensor(origin_flags, device=self.device, dtype=torch.bool)

        epoch = int(getattr(self, "epoch", 0))
        if self._step_epoch != epoch:
            self._step_epoch = epoch
            self._batch_in_epoch = 0
        self._current_step = epoch * len(self.train_loader) + self._batch_in_epoch + 1
        self._batch_in_epoch += 1
        self._apply_synchronized_hflip(batch)
        return batch

    def _apply_synchronized_hflip(self, batch):
        if self.hflip_prob <= 0:
            return
        scene_ids = batch["scene_ids"]
        for scene_id in torch.unique(scene_ids).detach().cpu().tolist():
            rng = random.Random(
                self.oamsc_seed
                + self._current_step * 1000003
                + int(scene_id) * 9176
            )
            if rng.random() >= self.hflip_prob:
                continue
            image_mask = scene_ids == int(scene_id)
            batch["img"][image_mask] = torch.flip(
                batch["img"][image_mask], dims=(-1,)
            )
            object_mask = image_mask[batch["batch_idx"].long()]
            batch["bboxes"][object_mask, 0] = (
                1.0 - batch["bboxes"][object_mask, 0]
            )

    def _ensure_hooks(self, model):
        if self._student_hook is None:
            detect = find_detect_module(model)

            def capture_student(_module, inputs):
                self._student_features["features"] = feature_list(inputs[0])

            self._student_hook = detect.register_forward_pre_hook(capture_student)
        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        if ema_model is not None and self._teacher_hook is None:
            detect = find_detect_module(ema_model)

            def capture_teacher(_module, inputs):
                self._teacher_features["features"] = feature_list(inputs[0])

            self._teacher_hook = detect.register_forward_pre_hook(capture_teacher)

    def _patch_loss(self, model):
        original_loss = model.loss

        def loss_with_oamsc(batch, preds=None):
            ema_model = getattr(getattr(self, "ema", None), "ema", None)
            if ema_model is None or "scene_ids" not in batch:
                return original_loss(batch, preds)
            self._ensure_hooks(model)
            self._student_features.clear()
            loss, loss_items = original_loss(batch, preds)
            student_features = self._student_features.get("features", [])
            origin_indices = torch.nonzero(batch["is_origin"], as_tuple=False).flatten()
            styled_indices = torch.nonzero(~batch["is_origin"], as_tuple=False).flatten()
            if not student_features or origin_indices.numel() == 0 or styled_indices.numel() == 0:
                return loss, loss_items

            self._teacher_features.clear()
            teacher = unwrap_model(ema_model)
            teacher.eval()
            with torch.no_grad():
                teacher(batch["img"][origin_indices])
            teacher_features = self._teacher_features.get("features", [])
            if not teacher_features:
                return loss, loss_items

            scene_ids = batch["scene_ids"]
            teacher_local_by_scene = {
                int(scene_ids[origin_index]): local_index
                for local_index, origin_index in enumerate(
                    origin_indices.detach().cpu().tolist()
                )
            }
            valid_styles = []
            teacher_rows = []
            for student_index in styled_indices.detach().cpu().tolist():
                local_index = teacher_local_by_scene.get(int(scene_ids[student_index]))
                if local_index is not None:
                    valid_styles.append(student_index)
                    teacher_rows.append(local_index)

            raw_global = loss.new_tensor(0.0)
            if valid_styles and self.global_weight > 0:
                student_global = global_embeddings(student_features)
                teacher_global = global_embeddings(teacher_features)
                if student_global is not None and teacher_global is not None:
                    student_rows = torch.tensor(
                        valid_styles, device=loss.device, dtype=torch.long
                    )
                    teacher_rows_tensor = torch.tensor(
                        teacher_rows, device=loss.device, dtype=torch.long
                    )
                    raw_global = (
                        1.0
                        - (
                            student_global[student_rows]
                            * teacher_global[teacher_rows_tensor].detach()
                        ).sum(dim=1)
                    ).mean()

            raw_roi = loss.new_tensor(0.0)
            if self.roi_weight > 0:
                pairs = build_object_pairs(
                    batch,
                    scene_ids,
                    origin_indices,
                    styled_indices,
                    self.max_objects,
                )
                if pairs:
                    student_roi = roi_embeddings(student_features, pairs, "student")
                    teacher_roi = roi_embeddings(teacher_features, pairs, "teacher")
                    if student_roi is not None and teacher_roi is not None:
                        raw_roi = (
                            1.0 - (student_roi * teacher_roi.detach()).sum(dim=1)
                        ).mean()

            ramp = ramp_weight(self._current_step, self.ramp_start, self.ramp_end)
            weighted = ramp * (
                self.global_weight * raw_global + self.roi_weight * raw_roi
            )
            # Ultralytics detection loss is summed over the local batch.
            loss = loss + weighted * int(batch["img"].shape[0])
            self._record_consistency(raw_global, raw_roi, ramp, weighted)
            return loss, loss_items

        model.loss = loss_with_oamsc

    def _record_consistency(self, raw_global, raw_roi, ramp, weighted):
        if RANK not in {-1, 0}:
            return
        frequency = max(int(getattr(self.args, "verbose", 1) and 20), 20)
        if self._current_step != 1 and self._current_step % frequency:
            return
        path = Path(self.save_dir) / "oamsc_loss.csv"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(
                    ["global_step", "epoch", "ramp", "global_raw", "roi_raw", "weighted"]
                )
            writer.writerow(
                [
                    self._current_step,
                    int(getattr(self, "epoch", 0)) + 1,
                    float(ramp),
                    float(raw_global.detach()),
                    float(raw_roi.detach()),
                    float(weighted.detach()),
                ]
            )

    def save_model(self):
        if self._teacher_hook is not None:
            self._teacher_hook.remove()
            self._teacher_hook = None
        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        if ema_model is not None:
            strip_oamsc_state(ema_model)
        return super().save_model()


OAMSCDetectionTrainer.__module__ = "train_yolov8x_oamsc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--global-weight", type=float, default=0.0)
    parser.add_argument("--roi-weight", type=float, default=0.0)
    parser.add_argument("--ramp-start", type=int, default=2000)
    parser.add_argument("--ramp-end", type=int, default=6000)
    parser.add_argument("--styles-per-anchor", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--enable-val", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    use_oamsc = args.global_weight > 0 or args.roi_weight > 0
    if use_oamsc and not args.scene_manifest:
        raise SystemExit("--scene-manifest is required when OAMSC losses are enabled.")
    if use_oamsc and args.batch % 2:
        raise SystemExit("Global batch size must be divisible across two GPUs.")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if str(script_dir) not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(script_dir) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )

    from ultralytics import YOLO

    model_source = args.resume if args.resume and args.resume.exists() else args.model
    model = YOLO(str(model_source))
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        optimizer="SGD",
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nbs=args.batch,
        val=args.enable_val,
        plots=False,
        patience=max(args.epochs, 1000),
        save_period=-1,
        amp=True,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=args.hflip_prob if not use_oamsc else 0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        close_mosaic=0,
    )
    if args.resume and args.resume.exists():
        train_kwargs["resume"] = str(args.resume)

    if use_oamsc:
        if DetectionTrainer is object:
            raise SystemExit("Ultralytics is required for OAMSC YOLO training.")
        os.environ["YOLO_OAMSC_SCENE_MANIFEST"] = str(args.scene_manifest.resolve())
        os.environ["YOLO_OAMSC_GLOBAL_WEIGHT"] = str(args.global_weight)
        os.environ["YOLO_OAMSC_ROI_WEIGHT"] = str(args.roi_weight)
        os.environ["YOLO_OAMSC_RAMP_START"] = str(args.ramp_start)
        os.environ["YOLO_OAMSC_RAMP_END"] = str(args.ramp_end)
        os.environ["YOLO_OAMSC_STYLES_PER_ANCHOR"] = str(args.styles_per_anchor)
        os.environ["YOLO_OAMSC_MAX_OBJECTS"] = str(args.max_objects)
        os.environ["YOLO_OAMSC_SEED"] = str(args.seed)
        os.environ["YOLO_OAMSC_HFLIP_PROB"] = str(args.hflip_prob)
        model.train(trainer=OAMSCDetectionTrainer, **train_kwargs)
    else:
        model.train(**train_kwargs)


if __name__ == "__main__":
    main()

"""Train and evaluate Faster R-CNN with optional OAMSC consistency.

This entry point implements scene-grouped sampling, synchronized geometry,
an EMA Origin teacher, global/GT-RoI alignment, diagnostics, and checkpoints.
All dataset and output roots are supplied through command-line arguments.
"""

from __future__ import print_function

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as NF
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import box_iou
from torchvision.transforms import functional as F


DEFAULT_RUN_NAME = "frcnn_r101_sgd"


class CocoStyleDetectionDataset(Dataset):
    """Load COCO boxes together with stable scene/style metadata.

    The metadata tensors let samplers and consistency losses reconstruct
    correspondence without matching proposals or inspecting filenames.
    """

    def __init__(self, annotation_json, train=False, hflip_prob=0.5):
        self.annotation_json = Path(annotation_json)
        self.train = train
        self.hflip_prob = float(hflip_prob)
        data = json.loads(self.annotation_json.read_text(encoding="utf-8"))
        self.images = data["images"]
        self.categories = data["categories"]
        scenes = sorted(set(str(image.get("scene", image["id"])) for image in self.images))
        styles = sorted(set(str(image.get("style", "")) for image in self.images))
        self.scene_to_id = {scene: idx for idx, scene in enumerate(scenes)}
        self.style_to_id = {style: idx for idx, style in enumerate(styles)}
        self.annotations_by_image = {image["id"]: [] for image in self.images}
        for ann in data["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image = Image.open(info["file_name"]).convert("RGB")
        width, height = image.size
        anns = self.annotations_by_image.get(info["id"], [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann["area"]))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]

        image_tensor = F.to_tensor(image)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(info["id"])]),
            "scene_id": torch.tensor([self.scene_to_id[str(info.get("scene", info["id"]))]], dtype=torch.int64),
            "style_id": torch.tensor([self.style_to_id[str(info.get("style", ""))]], dtype=torch.int64),
            "is_origin": torch.tensor(
                [int(str(info.get("style", "")).upper() == "ORIGIN")],
                dtype=torch.int64,
            ),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


class SceneGroupedBatchSampler(Sampler):
    """Build rank-local batches from complete same-scene view groups.

    In origin-anchor mode each group contains one Origin and a deterministic
    epoch-varying sample of styled views; groups are never split across ranks.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        seed=0,
        rank=0,
        world_size=1,
        origin_anchor=False,
        styles_per_anchor=3,
        repeat_threshold=0.0,
        repeat_cap=2,
        style_families=None,
    ):
        if batch_size < 2:
            raise ValueError("SceneGroupedBatchSampler requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_chunk_size = max(2, min(self.batch_size, self.batch_size // 2))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.origin_anchor = bool(origin_anchor)
        self.styles_per_anchor = int(styles_per_anchor)
        self.repeat_threshold = float(repeat_threshold)
        self.repeat_cap = max(1, int(repeat_cap))
        self.style_families = [
            tuple(str(style).strip() for style in family if str(style).strip())
            for family in (style_families or [])
        ]
        self.style_families = [family for family in self.style_families if family]
        self.style_to_family = {}
        for family_idx, family in enumerate(self.style_families):
            for style in family:
                key = style.casefold()
                if key in self.style_to_family:
                    raise ValueError(
                        "Style {} appears in more than one style family.".format(style)
                    )
                self.style_to_family[key] = family_idx
        self.epoch = 0
        self.groups = {}
        for idx, info in enumerate(dataset.images):
            scene = str(info.get("scene", info["id"]))
            self.groups.setdefault(scene, []).append(idx)
        self.scene_repeats = self._build_scene_repeats()
        self._length = len(self._batches_for_epoch(0, count_only=True))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batches_for_epoch(self, epoch, count_only=False):
        rng = random.Random(self.seed + int(epoch))
        scenes = []
        for scene in self.groups:
            scenes.extend([scene] * self.scene_repeats.get(scene, 1))
        rng.shuffle(scenes)
        chunk_rows = []
        for scene_position, scene in enumerate(scenes):
            indices = list(self.groups[scene])
            if self.origin_anchor:
                chunk_rows.append(
                    (
                        scene,
                        self._origin_anchor_chunk(
                            indices,
                            rng,
                            family_offset=int(epoch) + scene_position,
                        ),
                    )
                )
            else:
                rng.shuffle(indices)
                chunk_rows.extend((scene, chunk) for chunk in self._scene_chunks(indices))
        rng.shuffle(chunk_rows)
        chunk_rows.sort(key=lambda row: len(row[1]), reverse=True)

        batches = []
        batch_scenes = []
        for scene, chunk in chunk_rows:
            placed = False
            for batch_idx, batch in enumerate(batches):
                scene_conflict = self.origin_anchor and scene in batch_scenes[batch_idx]
                if not scene_conflict and len(batch) + len(chunk) <= self.batch_size:
                    batch.extend(chunk)
                    batch_scenes[batch_idx].add(scene)
                    placed = True
                    break
            if not placed:
                batches.append(list(chunk))
                batch_scenes.append({scene})

        if self.world_size > 1:
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[:usable]
        if count_only:
            return batches[self.rank::self.world_size]
        return batches[self.rank::self.world_size]

    def _build_scene_repeats(self):
        repeats = {scene: 1 for scene in self.groups}
        if self.repeat_threshold <= 0:
            return repeats

        category_scenes = {}
        scene_categories = {}
        for scene, indices in self.groups.items():
            labels = set()
            for ann in self.dataset.annotations_by_image.get(
                self.dataset.images[indices[0]]["id"], []
            ):
                labels.add(int(ann["category_id"]))
            scene_categories[scene] = labels
            for label in labels:
                category_scenes.setdefault(label, set()).add(scene)

        num_scenes = float(max(1, len(self.groups)))
        category_repeats = {}
        for label, label_scenes in category_scenes.items():
            frequency = len(label_scenes) / num_scenes
            category_repeats[label] = max(
                1.0,
                math.sqrt(self.repeat_threshold / max(frequency, 1e-12)),
            )

        for scene, labels in scene_categories.items():
            factor = max([category_repeats.get(label, 1.0) for label in labels] or [1.0])
            repeats[scene] = min(self.repeat_cap, max(1, int(math.ceil(factor))))
        return repeats

    def _origin_anchor_chunk(self, indices, rng, family_offset=0):
        origin = [
            idx
            for idx in indices
            if str(self.dataset.images[idx].get("style", "")).upper() == "ORIGIN"
        ]
        if len(origin) != 1:
            scene = self.dataset.images[indices[0]].get("scene", "unknown")
            raise RuntimeError(
                "Origin-anchored sampling requires exactly one ORIGIN image for scene {}."
                .format(scene)
            )
        variants = [idx for idx in indices if idx != origin[0]]
        take = min(self.styles_per_anchor, len(variants))
        if take < 1:
            scene = self.dataset.images[indices[0]].get("scene", "unknown")
            raise RuntimeError(
                "Origin-anchored sampling requires at least one style variant for scene {}."
                .format(scene)
            )

        if not self.style_families:
            rng.shuffle(variants)
            return [origin[0]] + variants[:take]

        variants_by_family = {idx: [] for idx in range(len(self.style_families))}
        ungrouped = []
        for idx in variants:
            style = str(self.dataset.images[idx].get("style", "")).casefold()
            family_idx = self.style_to_family.get(style)
            if family_idx is None:
                ungrouped.append(idx)
            else:
                variants_by_family[family_idx].append(idx)

        available_families = [
            family_idx
            for family_idx, family_variants in variants_by_family.items()
            if family_variants
        ]
        if len(available_families) < min(take, len(self.style_families)):
            scene = self.dataset.images[indices[0]].get("scene", "unknown")
            raise RuntimeError(
                "Scene {} does not contain enough configured style families: {} found, "
                "{} required.".format(
                    scene,
                    len(available_families),
                    min(take, len(self.style_families)),
                )
            )

        start = int(family_offset) % len(available_families)
        family_order = (
            available_families[start:] + available_families[:start]
        )
        selected = []
        for family_idx in family_order[:take]:
            selected.append(rng.choice(variants_by_family[family_idx]))

        if len(selected) < take:
            remaining = [
                idx for idx in variants if idx not in set(selected)
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[:take - len(selected)])
        return [origin[0]] + selected

    def _scene_chunks(self, indices):
        if len(indices) < 2:
            return []
        chunks = []
        start = 0
        while start < len(indices):
            remaining = len(indices) - start
            if remaining <= self.group_chunk_size:
                if remaining == 1:
                    if chunks:
                        chunks[-1].append(indices[start])
                    break
                chunks.append(indices[start:])
                break

            take = self.group_chunk_size
            if remaining - take == 1 and take > 2:
                take -= 1
            chunks.append(indices[start:start + take])
            start += take
        return chunks

    def __iter__(self):
        for batch in self._batches_for_epoch(self.epoch):
            yield batch

    def __len__(self):
        return self._length


class FixedSamplesBatchSampler(Sampler):
    """Sample a fixed number of random images per epoch across DDP ranks."""

    def __init__(
        self,
        dataset_size,
        batch_size,
        samples_per_epoch,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.dataset_size < 1:
            raise ValueError("FixedSamplesBatchSampler requires a non-empty dataset.")
        if not 1 <= self.samples_per_epoch <= self.dataset_size:
            raise ValueError(
                "samples_per_epoch must be in [1, {}], got {}.".format(
                    self.dataset_size, self.samples_per_epoch
                )
            )
        if self.samples_per_epoch % self.world_size != 0:
            raise ValueError(
                "samples_per_epoch must be divisible by world_size {}.".format(
                    self.world_size
                )
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected = rng.sample(range(self.dataset_size), self.samples_per_epoch)
        rank_indices = selected[self.rank::self.world_size]
        for start in range(0, len(rank_indices), self.batch_size):
            yield rank_indices[start:start + self.batch_size]

    def __len__(self):
        per_rank = self.samples_per_epoch // self.world_size
        return int(math.ceil(per_rank / float(self.batch_size)))


def collate_fn(batch):
    return tuple(zip(*batch))


class SceneSynchronizedCollate(object):
    """Apply one horizontal-flip decision to every view of a scene."""

    def __init__(self, hflip_prob=0.5):
        self.hflip_prob = float(hflip_prob)

    def __call__(self, batch):
        images, targets = tuple(zip(*batch))
        images = list(images)
        targets = list(targets)
        decisions = {}
        for target in targets:
            scene_id = int(target["scene_id"].item())
            if scene_id not in decisions:
                decisions[scene_id] = random.random() < self.hflip_prob

        for idx, (image, target) in enumerate(zip(images, targets)):
            if not decisions[int(target["scene_id"].item())]:
                continue
            width = image.shape[-1]
            images[idx] = torch.flip(image, dims=[-1])
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1
            targets[idx] = dict(target)
            targets[idx]["boxes"] = boxes
        return tuple(images), tuple(targets)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_is_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not distributed_is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not distributed_is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def setup_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return

    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise SystemExit("Distributed training requires CUDA.")

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed():
    if distributed_is_initialized():
        dist.destroy_process_group()


def broadcast_run_info(args, output_dir=None):
    if not args.distributed:
        return output_dir

    payload = [None]
    if is_main_process():
        payload[0] = {
            "output_dir": str(output_dir),
            "resolved_run_id": args.resolved_run_id,
        }
    dist.broadcast_object_list(payload, src=0)
    args.resolved_run_id = payload[0]["resolved_run_id"]
    return Path(payload[0]["output_dir"])


def reduce_epoch_loss(running_loss, steps, device):
    if not distributed_is_initialized():
        return running_loss / max(1, steps)

    values = torch.tensor([running_loss, float(steps)], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item() / max(1.0, values[1].item()))


def tensor_to_float(value):
    return float(value.detach().cpu())


def append_loss_components(path, row):
    fieldnames = [
        "epoch",
        "step",
        "global_step",
        "lr",
        "ramp_scale",
        "total_loss",
        "detection_loss",
        "consistency_loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_scene_feature_consistency",
        "loss_target_roi_feature_consistency",
        "loss_prediction_class_consistency",
        "loss_prediction_box_consistency",
    ]
    path = Path(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, 0.0) for key in fieldnames})


def write_loss_debug(output_dir, prefix, epoch, step, loss_value, loss_dict, image_ids, image_lookup):
    if output_dir is None:
        return None
    samples = []
    for image_id in image_ids:
        info = image_lookup.get(int(image_id), {})
        samples.append(
            {
                "image_id": int(image_id),
                "scene": info.get("scene"),
                "style": info.get("style"),
                "file_name": info.get("file_name"),
                "label_file": info.get("label_file"),
            }
        )
    payload = {
        "rank": get_rank(),
        "epoch": epoch,
        "step": step,
        "loss": loss_value,
        "loss_dict": {key: tensor_to_float(value) for key, value in loss_dict.items()},
        "samples": samples,
    }
    path = output_dir / "{}_rank{}_epoch{:03d}_step{:04d}.json".format(
        prefix, get_rank(), epoch, step
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_model(num_classes, trainable_layers, min_size, max_size, pretrained_backbone=True):
    from torchvision.models import ResNet101_Weights
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    try:
        backbone = resnet_fpn_backbone(
            "resnet101",
            weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
            trainable_layers=trainable_layers,
        )
    except TypeError:
        backbone = resnet_fpn_backbone(
            "resnet101",
            pretrained=pretrained_backbone,
            trainable_layers=trainable_layers,
        )

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )


class ModelEMA(object):
    """Maintain either an EMA state dictionary or a materialized teacher.

    Floating tensors use exponential averaging, while integer buffers are
    copied exactly to preserve valid detector state.
    """

    def __init__(self, model, decay=0.9998, materialize_model=False):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {}
        self.teacher = None
        if materialize_model:
            self.teacher = copy.deepcopy(model).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        else:
            self.copy_from(model)

    def copy_from(self, model):
        if self.teacher is not None:
            self.teacher.load_state_dict(model.state_dict(), strict=True)
            self.teacher.eval()
            return
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        model_state = model.state_dict()
        if self.teacher is not None:
            teacher_state = self.teacher.state_dict()
            for key, value in model_state.items():
                value = value.detach()
                if torch.is_floating_point(teacher_state[key]):
                    teacher_state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
                else:
                    teacher_state[key].copy_(value)
            return
        for key, value in model_state.items():
            value = value.detach()
            if key not in self.shadow:
                self.shadow[key] = value.clone()
            elif torch.is_floating_point(self.shadow[key]):
                self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        source = self.teacher.state_dict() if self.teacher is not None else self.shadow
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {
                key: value.detach().cpu()
                for key, value in source.items()
            },
        }

    def load_state_dict(self, state_dict):
        self.decay = float(state_dict.get("decay", self.decay))
        self.updates = int(state_dict.get("updates", 0))
        shadow = state_dict.get("shadow", state_dict)
        if self.teacher is not None:
            self.teacher.load_state_dict(shadow, strict=True)
            self.teacher.eval()
            return
        self.shadow = {
            key: value.detach().clone()
            for key, value in shadow.items()
        }

    def copy_to(self, model):
        state = self.teacher.state_dict() if self.teacher is not None else self.shadow
        model.load_state_dict(state, strict=True)

    def teacher_model(self):
        return self.teacher


def ema_model_state(ema_state):
    if isinstance(ema_state, dict) and "shadow" in ema_state:
        return ema_state["shadow"]
    return ema_state


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    device="cpu",
    weights="model",
):
    checkpoint = torch_load_checkpoint(path, device)
    if weights == "auto":
        weights = "ema" if checkpoint.get("model_ema") is not None else "model"
    if weights == "ema":
        if checkpoint.get("model_ema") is None:
            raise RuntimeError("Checkpoint has no EMA weights: {}".format(path))
        model.load_state_dict(ema_model_state(checkpoint["model_ema"]))
    else:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None:
        if checkpoint.get("model_ema") is not None:
            ema.load_state_dict(checkpoint["model_ema"])
        else:
            ema.copy_from(model)
    return int(checkpoint.get("epoch", 0))


def save_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": jsonable_args(args),
        },
        path,
    )


def evaluate_map(model, dataset, device, args):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise RuntimeError("torchmetrics detection mAP backend unavailable: {}".format(exc))

    was_training = model.training
    model.eval()
    old_score_thresh = getattr(model.roi_heads, "score_thresh", None)
    old_detections_per_img = getattr(model.roi_heads, "detections_per_img", None)
    model.roi_heads.score_thresh = args.val_score_threshold
    model.roi_heads.detections_per_img = args.val_detections_per_img

    loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
        sync_on_compute=False,
        max_detection_thresholds=[1, 10, args.val_metric_max_detections],
    )
    if hasattr(metric, "warn_on_many_detections"):
        metric.warn_on_many_detections = False

    pred_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)
            preds = []
            metric_targets = []
            for output in outputs:
                scores = output["scores"].detach().cpu()
                keep = scores >= args.val_score_threshold
                pred_count += int(keep.sum().item())
                preds.append(
                    {
                        "boxes": output["boxes"].detach().cpu()[keep],
                        "scores": scores[keep],
                        "labels": output["labels"].detach().cpu()[keep],
                    }
                )
            for target in targets:
                metric_targets.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )
            metric.update(preds, metric_targets)

    result = metric.compute()

    def metric_value(key):
        value = result.get(key)
        if value is None:
            return float("nan")
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = float(value)
        value = float(value)
        return float("nan") if value < 0 else value

    if old_score_thresh is not None:
        model.roi_heads.score_thresh = old_score_thresh
    if old_detections_per_img is not None:
        model.roi_heads.detections_per_img = old_detections_per_img
    if was_training:
        model.train()

    return {
        "images": len(dataset),
        "predictions": pred_count,
        "avg_predictions_per_image": pred_count / float(max(1, len(dataset))),
        "map50_95": metric_value("map"),
        "map50": metric_value("map_50"),
        "map75": metric_value("map_75"),
    }


def evaluate_map_distributed(model, dataset, device, args, epoch, output_dir):
    if not args.distributed:
        return evaluate_map(model, dataset, device, args)

    was_training = model.training
    model.eval()
    old_score_thresh = getattr(model.roi_heads, "score_thresh", None)
    old_detections_per_img = getattr(model.roi_heads, "detections_per_img", None)
    model.roi_heads.score_thresh = args.val_score_threshold
    model.roi_heads.detections_per_img = args.val_detections_per_img

    rank = get_rank()
    world_size = get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    preds_all = []
    targets_all = []
    pred_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)
            for output in outputs:
                scores = output["scores"].detach().cpu()
                keep = scores >= args.val_score_threshold
                pred_count += int(keep.sum().item())
                preds_all.append(
                    {
                        "boxes": output["boxes"].detach().cpu()[keep],
                        "scores": scores[keep],
                        "labels": output["labels"].detach().cpu()[keep],
                    }
                )
            for target in targets:
                targets_all.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )

    tmp_dir = Path(output_dir) / "val_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_path = tmp_dir / "epoch_{:03d}_rank_{:03d}.pt".format(epoch, rank)
    torch.save({"preds": preds_all, "targets": targets_all, "pred_count": pred_count}, part_path)
    dist.barrier()

    metrics = None
    if is_main_process():
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except Exception as exc:
            raise RuntimeError("torchmetrics detection mAP backend unavailable: {}".format(exc))

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=False,
            sync_on_compute=False,
            max_detection_thresholds=[1, 10, args.val_metric_max_detections],
        )
        if hasattr(metric, "warn_on_many_detections"):
            metric.warn_on_many_detections = False

        total_pred_count = 0
        for part_rank in range(world_size):
            part = torch.load(
                tmp_dir / "epoch_{:03d}_rank_{:03d}.pt".format(epoch, part_rank),
                map_location="cpu",
                weights_only=False,
            )
            if part["preds"]:
                metric.update(part["preds"], part["targets"])
            total_pred_count += int(part["pred_count"])

        result = metric.compute()

        def metric_value(key):
            value = result.get(key)
            if value is None:
                return float("nan")
            if hasattr(value, "detach"):
                value = value.detach().cpu()
                if value.numel() == 1:
                    value = float(value)
            value = float(value)
            return float("nan") if value < 0 else value

        metrics = {
            "images": len(dataset),
            "predictions": total_pred_count,
            "avg_predictions_per_image": total_pred_count / float(max(1, len(dataset))),
            "map50_95": metric_value("map"),
            "map50": metric_value("map_50"),
            "map75": metric_value("map_75"),
        }
        for part_rank in range(world_size):
            path = tmp_dir / "epoch_{:03d}_rank_{:03d}.pt".format(epoch, part_rank)
            if path.exists():
                path.unlink()

    dist.barrier()

    if old_score_thresh is not None:
        model.roi_heads.score_thresh = old_score_thresh
    if old_detections_per_img is not None:
        model.roi_heads.detections_per_img = old_detections_per_img
    if was_training:
        model.train()
    return metrics


def evaluate_with_optional_ema(model, ema, dataset, device, args, epoch, output_dir):
    use_ema = ema is not None and args.val_weights_type in ("auto", "ema")
    if not use_ema:
        return evaluate_map_distributed(model, dataset, device, args, epoch, output_dir)

    current_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    try:
        model.load_state_dict(ema_model_state(ema.state_dict()))
        return evaluate_map_distributed(model, dataset, device, args, epoch, output_dir)
    finally:
        model.load_state_dict(current_state)


def read_best_metric(val_log_path, metric_name):
    if not val_log_path.exists():
        return float("-inf")
    best = float("-inf")
    with val_log_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = float(row.get(metric_name, ""))
            except ValueError:
                continue
            if math.isfinite(value):
                best = max(best, value)
    return best


def safe_name(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    name = "".join(keep).strip("._-")
    return name or "run"


def run_dir_candidate(project, name, run_id):
    base_name = safe_name(name)
    if run_id:
        base_name = "{}_{}".format(base_name, safe_name(run_id))
    return Path(project) / base_name


def unique_run_dir(project, name, run_id=None):
    candidate = run_dir_candidate(project, name, run_id)
    if not candidate.exists():
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = candidate.with_name("{}_{}".format(candidate.name, timestamp))
    if not timestamped.exists():
        return timestamped

    for idx in range(2, 1000):
        numbered = candidate.with_name("{}_{}_{:02d}".format(candidate.name, timestamp, idx))
        if not numbered.exists():
            return numbered
    raise RuntimeError("Could not create a unique run directory under {}".format(project))


def read_json_or_none(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def short_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def load_selection_summary(data_path):
    return read_json_or_none(Path(data_path).parent / "selection_summary.json") or {}


def infer_sampling_strategy(data_path, summary):
    if summary.get("sampling_strategy"):
        return safe_name(summary["sampling_strategy"])

    parent = Path(data_path).parent.name
    if parent == "data_fixed_styles":
        return "fixed_styles_per_random_scene"
    if parent == "data":
        return "random_single_style_per_scene"
    return "custom_dataset"


def algorithm_id(data_path, summary):
    strategy = infer_sampling_strategy(data_path, summary)
    if strategy == "fixed_styles_per_random_scene":
        return "fixed"
    if strategy == "random_single_style_per_scene":
        return "unfixed"
    return safe_name(strategy)


def style_combo_id(dataset, summary):
    styles = summary.get("training_styles")
    if not styles:
        styles = sorted(set(image.get("style", "") for image in dataset.images if image.get("style", "")))
    styles = [safe_name(style) for style in styles if style]
    if not styles:
        return "unknown_styles"
    if len(styles) <= 4:
        return "{}styles_{}".format(len(styles), "_".join(styles))
    return "{}styles_{}".format(len(styles), short_hash(styles))


def automatic_run_id(args, dataset):
    summary = load_selection_summary(args.data)
    parts = [
        algorithm_id(args.data, summary),
        style_combo_id(dataset, summary),
    ]
    return safe_name("_".join(parts))


def resolve_output_dir(args, dataset):
    name = args.name or DEFAULT_RUN_NAME
    if args.resume is not None and args.name is None and args.run_id is None and not args.exist_ok:
        args.resolved_run_id = "resume"
        return args.resume.resolve().parent
    run_id = args.run_id
    if run_id is None and not args.no_auto_run_id:
        run_id = automatic_run_id(args, dataset)
    args.resolved_run_id = run_id
    if args.exist_ok:
        return run_dir_candidate(args.project, name, run_id)
    return unique_run_dir(args.project, name, run_id)


def jsonable_args(args):
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def write_run_config(output_dir, args, dataset, num_classes):
    style_counts = Counter(image.get("style", "") for image in dataset.images)
    annotation_count = sum(len(anns) for anns in dataset.annotations_by_image.values())
    selection_summary = load_selection_summary(args.data)
    config = {
        "output_dir": str(output_dir),
        "data": str(args.data.resolve()),
        "source_script": args.source_script or "manual",
        "resolved_run_id": args.resolved_run_id,
        "selection_summary": selection_summary,
        "num_images": len(dataset),
        "num_annotations": annotation_count,
        "num_classes_with_background": num_classes,
        "categories": dataset.categories,
        "style_counts": dict(sorted(style_counts.items())),
        "args": jsonable_args(args),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def set_warmup_lr(optimizer, base_lrs, step, warmup_iters, warmup_factor):
    if warmup_iters <= 0 or step > warmup_iters:
        return
    alpha = float(step) / float(warmup_iters)
    factor = warmup_factor * (1.0 - alpha) + alpha
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * factor


def feature_embeddings(features, feature_levels=None):
    """Pool selected FPN levels, concatenate channels, and L2-normalize."""

    if isinstance(features, torch.Tensor):
        if feature_levels:
            raise ValueError("Feature-level selection requires dictionary FPN features.")
        tensors = [features]
    elif isinstance(features, dict):
        if feature_levels:
            missing = [key for key in feature_levels if key not in features]
            if missing:
                raise KeyError(
                    "Requested FPN feature levels {} are unavailable; found {}."
                    .format(missing, list(features.keys()))
                )
            tensors = [features[key] for key in feature_levels]
        else:
            tensors = [features[key] for key in sorted(features.keys())]
    else:
        if feature_levels:
            raise ValueError("Feature-level selection requires dictionary FPN features.")
        tensors = list(features)
    pooled = []
    for value in tensors:
        pooled.append(value.float().mean(dim=(-2, -1)))
    return NF.normalize(torch.cat(pooled, dim=1), dim=1)


def scene_feature_consistency_loss(features, targets, feature_levels=None):
    embeddings = feature_embeddings(features, feature_levels)
    scene_ids = [int(target["scene_id"].item()) for target in targets]
    grouped = {}
    for idx, scene_id in enumerate(scene_ids):
        grouped.setdefault(scene_id, []).append(idx)

    losses = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        group = embeddings[indices]
        center = NF.normalize(group.mean(dim=0, keepdim=True), dim=1)
        losses.append((1.0 - (group * center).sum(dim=1)).mean())
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def object_roi_feature_consistency_loss(
    features,
    targets,
    image_sizes,
    box_roi_pool,
    max_objects_per_image=64,
):
    boxes_per_image = []
    roi_keys = []
    max_objects = int(max_objects_per_image)
    for image_idx, target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        if max_objects > 0:
            boxes = boxes[:max_objects]
            labels = labels[:max_objects]
        boxes_per_image.append(boxes)
        scene_id = int(target["scene_id"].item())
        for object_idx, label in enumerate(labels):
            roi_keys.append((scene_id, object_idx, int(label.item())))

    if not roi_keys:
        first_feature = next(iter(features.values())) if isinstance(features, dict) else features[0]
        return first_feature.sum() * 0.0

    roi_features = box_roi_pool(features, boxes_per_image, image_sizes)
    embeddings = NF.normalize(roi_features.float().mean(dim=(-2, -1)), dim=1)

    grouped = {}
    for idx, key in enumerate(roi_keys):
        grouped.setdefault(key, []).append(idx)

    losses = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        group = embeddings[indices]
        center = NF.normalize(group.mean(dim=0, keepdim=True), dim=1)
        losses.append((1.0 - (group * center).sum(dim=1)).mean())
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def consistency_ramp_scale(global_step, ramp_start, ramp_end):
    start = int(ramp_start)
    end = int(ramp_end)
    if end <= start:
        return 1.0
    if global_step <= start:
        return 0.0
    if global_step >= end:
        return 1.0
    return float(global_step - start) / float(end - start)


def origin_batch_indices(targets):
    by_scene = {}
    for idx, target in enumerate(targets):
        if int(target["is_origin"].item()) != 1:
            continue
        scene_id = int(target["scene_id"].item())
        if scene_id in by_scene:
            raise RuntimeError("Multiple ORIGIN samples found for scene {} in one batch.".format(scene_id))
        by_scene[scene_id] = idx
    scene_ids = sorted({int(target["scene_id"].item()) for target in targets})
    missing = [scene_id for scene_id in scene_ids if scene_id not in by_scene]
    if missing:
        raise RuntimeError(
            "Origin-teacher consistency requires an ORIGIN sample for every scene; missing {}."
            .format(missing)
        )
    return [by_scene[scene_id] for scene_id in scene_ids]


def origin_teacher_feature_consistency_loss(
    student_features,
    student_targets,
    teacher_features,
    teacher_targets,
    feature_levels=None,
):
    """Align each styled student embedding to its detached Origin teacher."""

    student_embeddings = feature_embeddings(student_features, feature_levels)
    teacher_embeddings = feature_embeddings(
        teacher_features, feature_levels
    ).detach()
    teacher_by_scene = {
        int(target["scene_id"].item()): teacher_embeddings[idx]
        for idx, target in enumerate(teacher_targets)
    }
    losses = []
    for idx, target in enumerate(student_targets):
        if int(target["is_origin"].item()) == 1:
            continue
        anchor = teacher_by_scene[int(target["scene_id"].item())]
        losses.append(1.0 - (student_embeddings[idx] * anchor).sum())
    if not losses:
        return student_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def origin_teacher_roi_feature_consistency_loss(
    student_features,
    student_targets,
    student_image_sizes,
    teacher_features,
    teacher_targets,
    teacher_image_sizes,
    box_roi_pool,
    max_objects_per_image=64,
):
    """Align corresponding GT-box RoIs between styled students and Origins.

    Keys combine scene, annotation order, and class, so no proposal or
    pseudo-label matching is introduced into the training objective.
    """

    max_objects = int(max_objects_per_image)
    teacher_boxes = []
    teacher_keys = []
    for target in teacher_targets:
        boxes = target["boxes"][:max_objects] if max_objects > 0 else target["boxes"]
        labels = target["labels"][:max_objects] if max_objects > 0 else target["labels"]
        teacher_boxes.append(boxes)
        scene_id = int(target["scene_id"].item())
        teacher_keys.extend(
            (scene_id, object_idx, int(label.item()))
            for object_idx, label in enumerate(labels)
        )

    if not teacher_keys:
        first_feature = next(iter(student_features.values()))
        return first_feature.sum() * 0.0

    with torch.no_grad():
        teacher_roi = box_roi_pool(teacher_features, teacher_boxes, teacher_image_sizes)
        teacher_embeddings = NF.normalize(
            teacher_roi.float().mean(dim=(-2, -1)), dim=1
        ).detach()
    teacher_by_key = {
        key: teacher_embeddings[idx]
        for idx, key in enumerate(teacher_keys)
    }

    student_boxes = []
    student_keys = []
    for target in student_targets:
        if int(target["is_origin"].item()) == 1:
            student_boxes.append(target["boxes"][:0])
            continue
        boxes = target["boxes"][:max_objects] if max_objects > 0 else target["boxes"]
        labels = target["labels"][:max_objects] if max_objects > 0 else target["labels"]
        student_boxes.append(boxes)
        scene_id = int(target["scene_id"].item())
        student_keys.extend(
            (scene_id, object_idx, int(label.item()))
            for object_idx, label in enumerate(labels)
        )

    if not student_keys:
        return teacher_embeddings.new_tensor(0.0)
    student_roi = box_roi_pool(student_features, student_boxes, student_image_sizes)
    student_embeddings = NF.normalize(student_roi.float().mean(dim=(-2, -1)), dim=1)

    losses = []
    for idx, key in enumerate(student_keys):
        anchor = teacher_by_key.get(key)
        if anchor is not None:
            losses.append(1.0 - (student_embeddings[idx] * anchor).sum())
    if not losses:
        return student_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def build_shared_teacher_proposals(
    teacher_model,
    image_list,
    features,
    targets,
    max_proposals,
    foreground_iou,
):
    rpn_was_training = teacher_model.rpn.training
    teacher_model.rpn.eval()
    try:
        with torch.no_grad():
            rpn_proposals, _ = teacher_model.rpn(image_list, features, None)
    finally:
        teacher_model.rpn.train(rpn_was_training)

    proposal_sets = []
    matched_labels = []
    limit = max(1, int(max_proposals))
    threshold = float(foreground_iou)
    for proposals, target in zip(rpn_proposals, targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        if gt_boxes.numel() == 0:
            proposal_sets.append(proposals[:0])
            matched_labels.append(gt_labels[:0])
            continue
        candidates = torch.cat([gt_boxes, proposals], dim=0)
        overlaps = box_iou(candidates, gt_boxes)
        best_iou, best_gt = overlaps.max(dim=1)
        keep = best_iou >= threshold
        candidates = candidates[keep][:limit]
        best_gt = best_gt[keep][:limit]
        proposal_sets.append(candidates.detach())
        matched_labels.append(gt_labels[best_gt].detach())
    return proposal_sets, matched_labels


def box_head_predictions(model, features, proposals, image_sizes):
    pooled = model.roi_heads.box_roi_pool(features, proposals, image_sizes)
    box_features = model.roi_heads.box_head(pooled)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    counts = [len(proposal) for proposal in proposals]
    return (
        class_logits.split(counts, dim=0),
        box_regression.split(counts, dim=0),
    )


def origin_teacher_prediction_consistency_losses(
    student_model,
    student_features,
    student_targets,
    student_image_sizes,
    teacher_model,
    teacher_features,
    teacher_targets,
    teacher_image_sizes,
    teacher_image_list,
    max_proposals=128,
    foreground_iou=0.5,
    temperature=2.0,
    teacher_confidence=0.0,
):
    teacher_proposals, matched_labels = build_shared_teacher_proposals(
        teacher_model,
        teacher_image_list,
        teacher_features,
        teacher_targets,
        max_proposals,
        foreground_iou,
    )
    with torch.no_grad():
        teacher_logits, teacher_box_regression = box_head_predictions(
            teacher_model,
            teacher_features,
            teacher_proposals,
            teacher_image_sizes,
        )
    teacher_by_scene = {}
    for idx, target in enumerate(teacher_targets):
        teacher_by_scene[int(target["scene_id"].item())] = (
            teacher_proposals[idx],
            matched_labels[idx],
            teacher_logits[idx].detach(),
            teacher_box_regression[idx].detach(),
        )

    student_proposals = []
    student_teacher_rows = []
    for target in student_targets:
        if int(target["is_origin"].item()) == 1:
            student_proposals.append(target["boxes"][:0])
            student_teacher_rows.append(None)
            continue
        teacher_row = teacher_by_scene[int(target["scene_id"].item())]
        student_proposals.append(teacher_row[0])
        student_teacher_rows.append(teacher_row)

    student_logits, student_box_regression = box_head_predictions(
        student_model,
        student_features,
        student_proposals,
        student_image_sizes,
    )

    cls_losses = []
    box_losses = []
    temp = float(temperature)
    for logits, box_regression, teacher_row in zip(
        student_logits, student_box_regression, student_teacher_rows
    ):
        if teacher_row is None or logits.numel() == 0:
            continue
        _, labels, anchor_logits, anchor_box_regression = teacher_row
        confidence = NF.softmax(anchor_logits.float(), dim=1)
        rows = torch.arange(len(labels), device=labels.device)
        keep = confidence[rows, labels] >= float(teacher_confidence)
        if not torch.any(keep):
            continue
        logits = logits[keep]
        box_regression = box_regression[keep]
        labels = labels[keep]
        anchor_logits = anchor_logits[keep]
        anchor_box_regression = anchor_box_regression[keep]
        cls_losses.append(
            NF.kl_div(
                NF.log_softmax(logits.float() / temp, dim=1),
                NF.softmax(anchor_logits.float() / temp, dim=1),
                reduction="batchmean",
            )
            * (temp * temp)
        )
        num_classes = logits.shape[1]
        student_deltas = box_regression.reshape(-1, num_classes, 4)
        anchor_deltas = anchor_box_regression.reshape(-1, num_classes, 4)
        rows = torch.arange(len(labels), device=labels.device)
        box_losses.append(
            NF.smooth_l1_loss(
                student_deltas[rows, labels],
                anchor_deltas[rows, labels],
                beta=1.0 / 9.0,
                reduction="mean",
            )
        )

    first_feature = next(iter(student_features.values()))
    zero = first_feature.sum() * 0.0
    cls_loss = torch.stack(cls_losses).mean() if cls_losses else zero
    box_loss = torch.stack(box_losses).mean() if box_losses else zero
    return cls_loss, box_loss


def subset_fpn_features(features, indices):
    if isinstance(features, dict):
        return {
            key: value[indices].detach()
            for key, value in features.items()
        }
    if isinstance(features, torch.Tensor):
        return features[indices].detach()
    return [value[indices].detach() for value in features]


def subset_image_list(image_list, image_sizes, indices):
    from torchvision.models.detection.image_list import ImageList

    return ImageList(
        image_list.tensors[indices].detach(),
        [image_sizes[idx] for idx in indices],
    )


def train_one_epoch(
    model,
    ema_model,
    optimizer,
    loader,
    device,
    scaler,
    ema,
    epoch,
    print_freq,
    clip_grad_norm,
    warmup_iters,
    warmup_factor,
    base_lrs,
    image_lookup,
    output_dir,
    debug_loss_threshold,
    consistency_weight,
    target_consistency_weight,
    target_consistency_max_objects,
    global_consistency_feature_levels=None,
    consistency_mode="centroid",
    consistency_ramp_start=0,
    consistency_ramp_end=0,
    prediction_class_consistency_weight=0.0,
    prediction_box_consistency_weight=0.0,
    prediction_teacher="ema",
    prediction_consistency_ramp_start=None,
    prediction_consistency_ramp_end=None,
    prediction_consistency_max_proposals=128,
    prediction_consistency_iou=0.5,
    prediction_consistency_temperature=2.0,
    prediction_teacher_confidence=0.0,
    global_step_offset=0,
    validation_callback=None,
    val_every_iters=0,
):
    """Run one epoch, attach OAMSC losses, update the student, then update EMA.

    Hooks expose transformed FPN tensors only during training and are removed
    before return, including error paths, so checkpoint inference is unchanged.
    """

    model.train()
    running_loss = 0.0
    start = time.time()
    captured = {}
    hook_handle = None
    transform_hook_handle = None
    use_prediction_consistency = (
        prediction_class_consistency_weight > 0
        or prediction_box_consistency_weight > 0
    )
    use_custom_consistency = (
        consistency_weight > 0
        or target_consistency_weight > 0
        or use_prediction_consistency
    )
    if use_custom_consistency:
        hook_model = model.module if hasattr(model, "module") else model

        def capture_backbone_features(_module, _inputs, output):
            captured["features"] = output

        def capture_transform_output(_module, _inputs, output):
            image_list, transformed_targets = output
            captured["image_list"] = image_list
            captured["image_sizes"] = image_list.image_sizes
            captured["targets"] = transformed_targets

        hook_handle = hook_model.backbone.register_forward_hook(capture_backbone_features)
        transform_hook_handle = hook_model.transform.register_forward_hook(capture_transform_output)

    try:
        for step, (images, targets) in enumerate(loader, 1):
            global_step = int(global_step_offset) + step
            if epoch == 1:
                set_warmup_lr(optimizer, base_lrs, step, warmup_iters, warmup_factor)

            image_ids = [int(target["image_id"].item()) for target in targets]
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in target.items()}
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            captured.clear()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                ramp_scale = consistency_ramp_scale(
                    global_step,
                    consistency_ramp_start,
                    consistency_ramp_end,
                )
                prediction_ramp_scale = consistency_ramp_scale(
                    global_step,
                    (
                        consistency_ramp_start
                        if prediction_consistency_ramp_start is None
                        else prediction_consistency_ramp_start
                    ),
                    (
                        consistency_ramp_end
                        if prediction_consistency_ramp_end is None
                        else prediction_consistency_ramp_end
                    ),
                )
                if consistency_mode == "centroid":
                    if consistency_weight > 0:
                        raw_consistency = scene_feature_consistency_loss(
                            captured["features"],
                            targets,
                            global_consistency_feature_levels,
                        )
                        loss_dict["loss_scene_feature_consistency"] = (
                            raw_consistency * consistency_weight * ramp_scale
                        )
                    if target_consistency_weight > 0:
                        hook_model = model.module if hasattr(model, "module") else model
                        raw_target_consistency = object_roi_feature_consistency_loss(
                            captured["features"],
                            captured["targets"],
                            captured["image_sizes"],
                            hook_model.roi_heads.box_roi_pool,
                            target_consistency_max_objects,
                        )
                        loss_dict["loss_target_roi_feature_consistency"] = (
                            raw_target_consistency
                            * target_consistency_weight
                            * ramp_scale
                        )
                elif use_custom_consistency:
                    hook_model = model.module if hasattr(model, "module") else model
                    origin_indices = origin_batch_indices(targets)
                    if consistency_mode == "origin_online":
                        teacher_model = hook_model
                        teacher_features = subset_fpn_features(
                            captured["features"],
                            origin_indices,
                        )
                        teacher_targets = [
                            captured["targets"][idx]
                            for idx in origin_indices
                        ]
                        teacher_image_list = subset_image_list(
                            captured["image_list"],
                            captured["image_sizes"],
                            origin_indices,
                        )
                    else:
                        teacher_model = (
                            ema.teacher_model() if ema is not None else None
                        )
                        if teacher_model is None:
                            raise RuntimeError(
                                "origin_teacher consistency requires --ema-teacher."
                            )
                        teacher_images = [images[idx] for idx in origin_indices]
                        teacher_input_targets = [
                            targets[idx] for idx in origin_indices
                        ]
                        teacher_model.eval()
                        with torch.no_grad():
                            teacher_image_list, teacher_targets = (
                                teacher_model.transform(
                                    teacher_images,
                                    teacher_input_targets,
                                )
                            )
                            teacher_features = teacher_model.backbone(
                                teacher_image_list.tensors
                            )

                    if consistency_weight > 0:
                        raw_consistency = origin_teacher_feature_consistency_loss(
                            captured["features"],
                            captured["targets"],
                            teacher_features,
                            teacher_targets,
                            global_consistency_feature_levels,
                        )
                        loss_dict["loss_scene_feature_consistency"] = (
                            raw_consistency * consistency_weight * ramp_scale
                        )
                    if target_consistency_weight > 0:
                        raw_target_consistency = (
                            origin_teacher_roi_feature_consistency_loss(
                                captured["features"],
                                captured["targets"],
                                captured["image_sizes"],
                                teacher_features,
                                teacher_targets,
                                teacher_image_list.image_sizes,
                                hook_model.roi_heads.box_roi_pool,
                                target_consistency_max_objects,
                            )
                        )
                        loss_dict["loss_target_roi_feature_consistency"] = (
                            raw_target_consistency
                            * target_consistency_weight
                            * ramp_scale
                        )
                    if use_prediction_consistency:
                        prediction_teacher_model = teacher_model
                        prediction_teacher_features = teacher_features
                        prediction_teacher_targets = teacher_targets
                        prediction_teacher_image_sizes = (
                            teacher_image_list.image_sizes
                        )
                        prediction_teacher_image_list = teacher_image_list
                        if (
                            consistency_mode == "origin_online"
                            or prediction_teacher == "online_origin"
                        ):
                            prediction_teacher_model = hook_model
                            prediction_teacher_features = subset_fpn_features(
                                captured["features"],
                                origin_indices,
                            )
                            prediction_teacher_targets = [
                                captured["targets"][idx]
                                for idx in origin_indices
                            ]
                            prediction_teacher_image_sizes = [
                                captured["image_sizes"][idx]
                                for idx in origin_indices
                            ]
                            prediction_teacher_image_list = subset_image_list(
                                captured["image_list"],
                                captured["image_sizes"],
                                origin_indices,
                            )
                        raw_class_consistency, raw_box_consistency = (
                            origin_teacher_prediction_consistency_losses(
                                hook_model,
                                captured["features"],
                                captured["targets"],
                                captured["image_sizes"],
                                prediction_teacher_model,
                                prediction_teacher_features,
                                prediction_teacher_targets,
                                prediction_teacher_image_sizes,
                                prediction_teacher_image_list,
                                prediction_consistency_max_proposals,
                                prediction_consistency_iou,
                                prediction_consistency_temperature,
                                prediction_teacher_confidence,
                            )
                        )
                        if prediction_class_consistency_weight > 0:
                            loss_dict["loss_prediction_class_consistency"] = (
                                raw_class_consistency
                                * prediction_class_consistency_weight
                                * prediction_ramp_scale
                            )
                        if prediction_box_consistency_weight > 0:
                            loss_dict["loss_prediction_box_consistency"] = (
                                raw_box_consistency
                                * prediction_box_consistency_weight
                                * prediction_ramp_scale
                            )
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                debug_path = write_loss_debug(
                    output_dir,
                    "nonfinite_loss",
                    epoch,
                    step,
                    tensor_to_float(losses),
                    loss_dict,
                    image_ids,
                    image_lookup,
                )
                raise RuntimeError(
                    "Non-finite loss at epoch {}, step {}: {}. Debug batch: {}".format(
                        epoch, step, losses, debug_path
                    )
                )

            loss_value = float(losses.detach().cpu())
            if debug_loss_threshold > 0 and loss_value >= debug_loss_threshold:
                write_loss_debug(
                    output_dir,
                    "high_loss",
                    epoch,
                    step,
                    loss_value,
                    loss_dict,
                    image_ids,
                    image_lookup,
                )

            if scaler is not None:
                scaler.scale(losses).backward()
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            if ema is not None:
                ema.update(ema_model)

            running_loss += loss_value
            if is_main_process() and (step == 1 or step % print_freq == 0 or step == len(loader)):
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = running_loss / step
                elapsed = time.time() - start
                component_values = {
                    key: tensor_to_float(value)
                    for key, value in loss_dict.items()
                }
                detection_keys = [
                    "loss_classifier",
                    "loss_box_reg",
                    "loss_objectness",
                    "loss_rpn_box_reg",
                ]
                consistency_keys = [
                    "loss_scene_feature_consistency",
                    "loss_target_roi_feature_consistency",
                    "loss_prediction_class_consistency",
                    "loss_prediction_box_consistency",
                ]
                append_loss_components(
                    Path(output_dir) / "loss_components.csv",
                    {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "lr": lr,
                        "ramp_scale": ramp_scale,
                        "total_loss": loss_value,
                        "detection_loss": sum(
                            component_values.get(key, 0.0) for key in detection_keys
                        ),
                        "consistency_loss": sum(
                            component_values.get(key, 0.0) for key in consistency_keys
                        ),
                        **component_values,
                    },
                )
                print(
                    "epoch {:03d} step {:04d}/{:04d} loss {:.4f} avg_loss {:.4f} lr {:.6g} time {:.1f}s".format(
                        epoch, step, len(loader), loss_value, avg_loss, lr, elapsed
                    )
                )

            if validation_callback is not None and val_every_iters > 0 and global_step % val_every_iters == 0:
                validation_callback(epoch, step, global_step)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if transform_hook_handle is not None:
            transform_hook_handle.remove()
    return reduce_epoch_loss(running_loss, len(loader), device)


def parse_style_family_groups(value):
    groups = []
    for raw_group in str(value or "").split(";"):
        group = [
            style.strip()
            for style in raw_group.split(",")
            if style.strip()
        ]
        if group:
            groups.append(tuple(group))
    return tuple(groups)


def parse_feature_levels(value):
    return tuple(
        level.strip()
        for level in str(value or "").split(",")
        if level.strip()
    )


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=project_root / "data" / "train_coco.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--epochs", type=int, default=54)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--project", type=Path, default=project_root / "runs" / "train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--source-script", default=None)
    parser.add_argument("--no-auto-run-id", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--warmup-factor", type=float, default=0.001)
    parser.add_argument("--debug-loss-threshold", type=float, default=100.0)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--target-consistency-weight", type=float, default=0.0)
    parser.add_argument("--target-consistency-max-objects", type=int, default=64)
    parser.add_argument("--scene-grouped-batches", action="store_true")
    parser.add_argument(
        "--random-samples-per-epoch",
        type=int,
        default=0,
        help=(
            "When scene grouping is disabled, randomly sample this many global "
            "images per epoch. Zero traverses the full dataset."
        ),
    )
    parser.add_argument("--origin-anchor-groups", action="store_true")
    parser.add_argument("--styles-per-anchor", type=int, default=3)
    parser.add_argument(
        "--style-family-groups",
        default="",
        help=(
            "Semicolon-separated style families with comma-separated styles. "
            "When set, one anchored batch samples styles from distinct families."
        ),
    )
    parser.add_argument("--sync-scene-hflip", action="store_true")
    parser.add_argument("--scene-repeat-threshold", type=float, default=0.0)
    parser.add_argument("--scene-repeat-cap", type=int, default=2)
    parser.add_argument(
        "--consistency-mode",
        choices=["centroid", "origin_teacher", "origin_online"],
        default="centroid",
    )
    parser.add_argument("--ema-teacher", action="store_true")
    parser.add_argument(
        "--global-consistency-feature-levels",
        default="",
        help="Comma-separated FPN keys for global consistency; empty uses every level.",
    )
    parser.add_argument("--consistency-ramp-start", type=int, default=0)
    parser.add_argument("--consistency-ramp-end", type=int, default=0)
    parser.add_argument("--prediction-class-consistency-weight", type=float, default=0.0)
    parser.add_argument("--prediction-box-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--prediction-teacher",
        choices=["ema", "online_origin"],
        default="ema",
    )
    parser.add_argument("--prediction-consistency-ramp-start", type=int, default=-1)
    parser.add_argument("--prediction-consistency-ramp-end", type=int, default=-1)
    parser.add_argument("--prediction-consistency-max-proposals", type=int, default=128)
    parser.add_argument("--prediction-consistency-iou", type=float, default=0.5)
    parser.add_argument("--prediction-consistency-temperature", type=float, default=2.0)
    parser.add_argument("--prediction-teacher-confidence", type=float, default=0.0)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-every-iters", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--val-score-threshold", type=float, default=0.001)
    parser.add_argument("--val-detections-per-img", type=int, default=300)
    parser.add_argument("--val-metric-max-detections", type=int, default=100)
    parser.add_argument("--val-weights-type", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument("--best-metric", choices=["map50_95", "map50"], default="map50_95")
    return parser.parse_args()


def main():
    args = parse_args()
    style_families = parse_style_family_groups(args.style_family_groups)
    global_feature_levels = parse_feature_levels(
        args.global_consistency_feature_levels
    )
    setup_distributed(args)

    if args.prepare or not args.data.exists():
        # Public archive entry points are colocated in the code directory.
        prepare_script = Path(__file__).resolve().with_name(
            "prepare_fasterrcnn_cityscapes20.py"
        )
        import subprocess
        import sys

        if is_main_process():
            subprocess.check_call([sys.executable, str(prepare_script), "--seed", str(args.seed)])
        if args.distributed:
            dist.barrier()

    seed_everything(args.seed + get_rank())

    if args.distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available to PyTorch. Install a CUDA-enabled torch build or pass --allow-cpu for a smoke test."
        )
    if device.type == "cuda" and is_main_process():
        print("CUDA device count: {}".format(torch.cuda.device_count()))
        print("training processes: {}".format(get_world_size()))
        print("CUDA device rank0: {}".format(torch.cuda.get_device_name(0)))

    if args.origin_anchor_groups and not args.scene_grouped_batches:
        raise SystemExit("--origin-anchor-groups requires --scene-grouped-batches.")
    if args.scene_grouped_batches and args.random_samples_per_epoch > 0:
        raise SystemExit(
            "--random-samples-per-epoch cannot be combined with "
            "--scene-grouped-batches."
        )
    if args.random_samples_per_epoch < 0:
        raise SystemExit("--random-samples-per-epoch must be non-negative.")
    if args.origin_anchor_groups and args.styles_per_anchor < 1:
        raise SystemExit("--styles-per-anchor must be at least 1.")
    if style_families and not args.origin_anchor_groups:
        raise SystemExit("--style-family-groups requires --origin-anchor-groups.")
    if style_families and len(style_families) < args.styles_per_anchor:
        raise SystemExit(
            "--style-family-groups defines {} families, but {} styles per anchor "
            "were requested.".format(len(style_families), args.styles_per_anchor)
        )
    if args.consistency_mode in ("origin_teacher", "origin_online"):
        if not args.origin_anchor_groups:
            raise SystemExit(
                "{} consistency requires --origin-anchor-groups."
                .format(args.consistency_mode)
            )
        if args.consistency_mode == "origin_teacher" and not args.ema_teacher:
            raise SystemExit("origin_teacher consistency requires --ema-teacher.")
        if args.consistency_mode == "origin_online" and not args.no_ema:
            raise SystemExit("origin_online consistency requires --no-ema.")
    if (
        args.prediction_class_consistency_weight > 0
        or args.prediction_box_consistency_weight > 0
    ) and args.consistency_mode not in ("origin_teacher", "origin_online"):
        raise SystemExit(
            "Prediction consistency requires origin_teacher or origin_online mode."
        )
    if not 0.0 <= args.prediction_teacher_confidence <= 1.0:
        raise SystemExit("--prediction-teacher-confidence must be in [0, 1].")

    dataset_hflip = 0.0 if args.sync_scene_hflip else args.hflip_prob
    dataset = CocoStyleDetectionDataset(args.data, train=True, hflip_prob=dataset_hflip)
    if style_families:
        configured_styles = {
            style.casefold()
            for family in style_families
            for style in family
        }
        dataset_styles = {
            str(image.get("style", "")).casefold()
            for image in dataset.images
            if str(image.get("style", "")).upper() != "ORIGIN"
        }
        missing_styles = sorted(configured_styles - dataset_styles)
        ungrouped_styles = sorted(dataset_styles - configured_styles)
        if missing_styles or ungrouped_styles:
            raise SystemExit(
                "Style-family mismatch. Missing configured styles: {}; ungrouped "
                "dataset styles: {}.".format(missing_styles, ungrouped_styles)
            )
    val_dataset = None
    if args.val_data is not None:
        if not args.val_data.exists():
            raise SystemExit("Validation COCO json not found: {}".format(args.val_data))
        val_dataset = CocoStyleDetectionDataset(args.val_data, train=False)
    image_lookup = {int(image["id"]): image for image in dataset.images}
    num_classes = len(dataset.categories) + 1
    world_size = get_world_size()
    if args.batch_size % world_size != 0:
        raise SystemExit(
            "--batch-size is the global batch size and must be divisible by world size {}. Got {}.".format(
                world_size, args.batch_size
            )
        )
    per_process_batch_size = args.batch_size // world_size
    sampler = None
    if args.scene_grouped_batches:
        if per_process_batch_size < 2:
            raise SystemExit("--scene-grouped-batches requires per-process batch size >= 2.")
        anchor_group_size = 1 + args.styles_per_anchor
        if args.origin_anchor_groups and anchor_group_size > per_process_batch_size:
            raise SystemExit(
                "ORIGIN anchor group size {} exceeds per-process batch size {}."
                .format(anchor_group_size, per_process_batch_size)
            )
        sampler = SceneGroupedBatchSampler(
            dataset,
            per_process_batch_size,
            seed=args.seed,
            rank=get_rank(),
            world_size=world_size,
            origin_anchor=args.origin_anchor_groups,
            styles_per_anchor=args.styles_per_anchor,
            repeat_threshold=args.scene_repeat_threshold,
            repeat_cap=args.scene_repeat_cap,
            style_families=style_families,
        )
        train_collate = (
            SceneSynchronizedCollate(args.hflip_prob)
            if args.sync_scene_hflip
            else collate_fn
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=train_collate,
        )
    elif args.random_samples_per_epoch > 0:
        sampler = FixedSamplesBatchSampler(
            len(dataset),
            per_process_batch_size,
            args.random_samples_per_epoch,
            seed=args.seed,
            rank=get_rank(),
            world_size=world_size,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None
        loader = DataLoader(
            dataset,
            batch_size=per_process_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )

    model = build_model(
        num_classes=num_classes,
        trainable_layers=args.trainable_layers,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.to(device)
    ema = None
    if not args.no_ema:
        ema = ModelEMA(
            model,
            decay=args.ema_decay,
            materialize_model=args.ema_teacher,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    milestones = [max(1, int(args.epochs * 0.67)), max(2, int(args.epochs * 0.89))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = None
    if device.type == "cuda" and not args.no_amp:
        scaler = torch.cuda.amp.GradScaler()

    output_dir = None
    if is_main_process():
        output_dir = resolve_output_dir(args, dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = broadcast_run_info(args, output_dir)
    args.output_dir = output_dir
    if is_main_process():
        write_run_config(output_dir, args, dataset, num_classes)
    log_path = output_dir / "train_log.csv"
    val_log_path = output_dir / "val_log.csv"
    best_metric_value = read_best_metric(val_log_path, args.best_metric) if is_main_process() else float("-inf")

    start_epoch = 1
    if args.resume is not None:
        last_epoch = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            ema=ema,
            device=device,
            weights="model",
        )
        start_epoch = last_epoch + 1
        if is_main_process():
            print("Resumed from {} at epoch {}".format(args.resume, last_epoch))

    train_model = model
    if args.distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    f = None
    writer = None
    if is_main_process():
        f = log_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr"])
        if f.tell() == 0:
            writer.writeheader()

        total_iters = args.epochs * len(loader)
        print("images: {}".format(len(dataset)))
        print("classes including background: {}".format(num_classes))
        print("epochs: {}".format(args.epochs))
        print("global_batch_size: {}".format(args.batch_size))
        print("per_process_batch_size: {}".format(per_process_batch_size))
        print("total_iterations: {}".format(total_iters))
        print("optimizer: SGD")
        print("backbone: ResNet101 ImageNet pretrained + FPN")
        print("ema: {}".format("enabled decay={}".format(args.ema_decay) if ema is not None else "disabled"))
        print("warmup: iters={} factor={}".format(args.warmup_iters, args.warmup_factor))
        print("clip_grad_norm: {}".format(args.clip_grad_norm))
        print("debug_loss_threshold: {}".format(args.debug_loss_threshold))
        print("hflip_prob: {}".format(args.hflip_prob))
        print("scene_grouped_batches: {}".format(args.scene_grouped_batches))
        print("origin_anchor_groups: {}".format(args.origin_anchor_groups))
        print("styles_per_anchor: {}".format(args.styles_per_anchor))
        print(
            "style_family_groups: {}"
            .format(args.style_family_groups if style_families else "disabled")
        )
        print("sync_scene_hflip: {}".format(args.sync_scene_hflip))
        print("scene_repeat_threshold: {}".format(args.scene_repeat_threshold))
        print("scene_repeat_cap: {}".format(args.scene_repeat_cap))
        print("consistency_mode: {}".format(args.consistency_mode))
        print("ema_teacher: {}".format(args.ema_teacher))
        print(
            "global_consistency_feature_levels: {}"
            .format(
                ",".join(global_feature_levels)
                if global_feature_levels
                else "all"
            )
        )
        print(
            "consistency_ramp: {} -> {}".format(
                args.consistency_ramp_start,
                args.consistency_ramp_end,
            )
        )
        print("consistency_weight: {}".format(args.consistency_weight))
        print("target_consistency_weight: {}".format(args.target_consistency_weight))
        print("target_consistency_max_objects: {}".format(args.target_consistency_max_objects))
        print(
            "prediction_class_consistency_weight: {}"
            .format(args.prediction_class_consistency_weight)
        )
        print(
            "prediction_box_consistency_weight: {}"
            .format(args.prediction_box_consistency_weight)
        )
        print("prediction_teacher: {}".format(args.prediction_teacher))
        print(
            "prediction_consistency_ramp: {} -> {}"
            .format(
                args.prediction_consistency_ramp_start,
                args.prediction_consistency_ramp_end,
            )
        )
        print(
            "prediction_teacher_confidence: {}"
            .format(args.prediction_teacher_confidence)
        )
        print("val_data: {}".format(args.val_data if args.val_data is not None else "disabled"))
        if val_dataset is not None:
            print("val_images: {}".format(len(val_dataset)))
            print("val_every: {}".format(args.val_every))
            print("val_every_iters: {}".format(args.val_every_iters))
            print("val_weights_type: {}".format(args.val_weights_type))
            print("best_metric: {}".format(args.best_metric))
            print("best_metric_so_far: {}".format(best_metric_value))
        print("output_dir: {}".format(output_dir))
        print("run_id: {}".format(args.resolved_run_id))

    def run_validation(epoch, step, global_step, trigger):
        nonlocal best_metric_value
        val_metrics = evaluate_with_optional_ema(model, ema, val_dataset, device, args, epoch, output_dir)
        if is_main_process():
            val_fieldnames = [
                "epoch",
                "step",
                "global_step",
                "trigger",
                "weights_type",
                "best_metric",
                "is_best",
                "images",
                "predictions",
                "avg_predictions_per_image",
                "map50_95",
                "map50",
                "map75",
            ]
            current_metric = val_metrics[args.best_metric]
            is_best = math.isfinite(current_metric) and current_metric > best_metric_value
            if is_best:
                best_metric_value = current_metric
                save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, scaler, ema, epoch, args)
            with val_log_path.open("a", newline="", encoding="utf-8") as vf:
                val_writer = csv.DictWriter(vf, fieldnames=val_fieldnames)
                if vf.tell() == 0:
                    val_writer.writeheader()
                row = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "trigger": trigger,
                    "weights_type": args.val_weights_type,
                    "best_metric": args.best_metric,
                    "is_best": int(is_best),
                }
                row.update(val_metrics)
                val_writer.writerow(row)
            print(
                "epoch {:03d} step {:04d} global_step {:06d} val map50_95 {:.4f} map50 {:.4f} best_{} {:.4f}{}".format(
                    epoch,
                    step,
                    global_step,
                    val_metrics["map50_95"],
                    val_metrics["map50"],
                    args.best_metric,
                    best_metric_value,
                    " saved best.pth" if is_best else "",
                )
            )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            global_step_offset = (epoch - 1) * len(loader)
            avg_loss = train_one_epoch(
                train_model,
                model,
                optimizer,
                loader,
                device,
                scaler,
                ema,
                epoch,
                args.print_freq,
                args.clip_grad_norm,
                args.warmup_iters,
                args.warmup_factor,
                base_lrs,
                image_lookup,
                output_dir,
                args.debug_loss_threshold,
                args.consistency_weight,
                args.target_consistency_weight,
                args.target_consistency_max_objects,
                global_consistency_feature_levels=global_feature_levels,
                consistency_mode=args.consistency_mode,
                consistency_ramp_start=args.consistency_ramp_start,
                consistency_ramp_end=args.consistency_ramp_end,
                prediction_class_consistency_weight=(
                    args.prediction_class_consistency_weight
                ),
                prediction_box_consistency_weight=(
                    args.prediction_box_consistency_weight
                ),
                prediction_teacher=args.prediction_teacher,
                prediction_consistency_ramp_start=(
                    None
                    if args.prediction_consistency_ramp_start < 0
                    else args.prediction_consistency_ramp_start
                ),
                prediction_consistency_ramp_end=(
                    None
                    if args.prediction_consistency_ramp_end < 0
                    else args.prediction_consistency_ramp_end
                ),
                prediction_consistency_max_proposals=(
                    args.prediction_consistency_max_proposals
                ),
                prediction_consistency_iou=args.prediction_consistency_iou,
                prediction_consistency_temperature=(
                    args.prediction_consistency_temperature
                ),
                prediction_teacher_confidence=(
                    args.prediction_teacher_confidence
                ),
                global_step_offset=global_step_offset,
                validation_callback=(
                    lambda val_epoch, val_step, val_global_step: run_validation(
                        val_epoch, val_step, val_global_step, "iteration"
                    )
                )
                if val_dataset is not None and args.val_every_iters > 0
                else None,
                val_every_iters=args.val_every_iters,
            )
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            if is_main_process():
                writer.writerow({"epoch": epoch, "loss": avg_loss, "lr": lr})
                f.flush()

                save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, scaler, ema, epoch, args)
                if args.save_period > 0 and epoch % args.save_period == 0:
                    save_checkpoint(
                        output_dir / "epoch_{:03d}.pth".format(epoch),
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        epoch,
                        args,
                    )

            if (
                val_dataset is not None
                and args.val_every_iters <= 0
                and args.val_every > 0
                and epoch % args.val_every == 0
            ):
                run_validation(epoch, len(loader), epoch * len(loader), "epoch")
            if args.distributed:
                dist.barrier()
    finally:
        if f is not None:
            f.close()

    if is_main_process():
        print("Training complete. Last checkpoint: {}".format(output_dir / "last.pth"))
    cleanup_distributed()


if __name__ == "__main__":
    main()

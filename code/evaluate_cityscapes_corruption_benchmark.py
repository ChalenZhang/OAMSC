"""Run the fixed Cityscapes Corruption benchmark from clean source images.

The implementation covers 15 corruptions, five severities, deterministic
seeding, optional sharding, COCO evaluation, and CSV/JSON summaries.
"""

from __future__ import print_function

import argparse
import csv
import inspect
import io
import json
import math
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from evaluate_bdd100k_all import (
    CLASS_IDS,
    CLASS_NAMES,
    ap50_from_extended_summary,
    metric_value,
    per_class_values,
)
from train_fasterrcnn_resnet101 import build_model, collate_fn, load_checkpoint

try:
    from imagecorruptions import corrupt as imagecorruptions_corrupt
    import imagecorruptions.corruptions as imagecorruptions_corruptions
except Exception:
    imagecorruptions_corrupt = None
    imagecorruptions_corruptions = None


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "jpeg_compression",
    "pixelate",
]

DISPLAY_NAMES = {
    "gaussian_noise": "Gauss.",
    "shot_noise": "Shot",
    "impulse_noise": "Impulse",
    "defocus_blur": "Defocus",
    "glass_blur": "Glass",
    "motion_blur": "Motion",
    "zoom_blur": "Zoom",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Bright",
    "contrast": "Contrast",
    "elastic_transform": "Elastic",
    "jpeg_compression": "JPEG",
    "pixelate": "Pixel",
}

SUMMARY_COLUMNS = [
    "run",
    "Clean",
    "Gauss.",
    "Shot",
    "Impulse",
    "Defocus",
    "Glass",
    "Motion",
    "Zoom",
    "Snow",
    "Frost",
    "Fog",
    "Bright",
    "Contrast",
    "Elastic",
    "JPEG",
    "Pixel",
    "mPC",
    "rPC",
]


def stable_seed(*values):
    text = "|".join(str(value) for value in values)
    seed = 0
    for char in text:
        seed = (seed * 131 + ord(char)) % (2 ** 32)
    return seed


def as_array(image):
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def to_image(array):
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def gaussian_noise(image, severity, rng):
    sigmas = [0.08, 0.12, 0.18, 0.26, 0.38]
    x = as_array(image)
    x = x + rng.normal(0.0, sigmas[severity - 1], x.shape)
    return to_image(x)


def shot_noise(image, severity, rng):
    scales = [60, 25, 12, 5, 3]
    x = as_array(image)
    scale = scales[severity - 1]
    x = rng.poisson(np.clip(x, 0, 1) * scale) / float(scale)
    return to_image(x)


def impulse_noise(image, severity, rng):
    amounts = [0.03, 0.06, 0.09, 0.17, 0.27]
    x = as_array(image)
    mask = rng.random(x.shape[:2])
    amount = amounts[severity - 1]
    salt = mask < amount / 2.0
    pepper = (mask >= amount / 2.0) & (mask < amount)
    x[salt] = 1.0
    x[pepper] = 0.0
    return to_image(x)


def defocus_blur(image, severity, rng):
    radii = [1.0, 1.5, 2.0, 2.8, 4.0]
    return image.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))


def glass_blur(image, severity, rng):
    x = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=0.6 + 0.35 * severity))).copy()
    h, w = x.shape[:2]
    max_delta = severity
    iterations = [1, 2, 2, 3, 4][severity - 1]
    for _ in range(iterations):
        for y in range(max_delta, h - max_delta, 2):
            for x0 in range(max_delta, w - max_delta, 2):
                dy = int(rng.integers(-max_delta, max_delta + 1))
                dx = int(rng.integers(-max_delta, max_delta + 1))
                yy = y + dy
                xx = x0 + dx
                x[y, x0], x[yy, xx] = x[yy, xx].copy(), x[y, x0].copy()
    return Image.fromarray(x, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.4))


def motion_blur(image, severity, rng):
    sizes = [5, 7, 9, 13, 17]
    size = sizes[severity - 1]
    x = np.asarray(image, dtype=np.float32)
    pad = size // 2
    if severity % 2 == 0:
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(x)
        for offset in range(size):
            out += padded[:, offset : offset + x.shape[1], :]
    else:
        padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(x)
        for offset in range(size):
            out += padded[offset : offset + x.shape[0], offset : offset + x.shape[1], :]
    out /= float(size)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def zoom_blur(image, severity, rng):
    x = as_array(image)
    h, w = x.shape[:2]
    zooms = {
        1: [1.02, 1.04],
        2: [1.03, 1.06, 1.09],
        3: [1.04, 1.08, 1.12, 1.16],
        4: [1.05, 1.10, 1.15, 1.20, 1.25],
        5: [1.06, 1.12, 1.18, 1.24, 1.30],
    }[severity]
    out = x.copy()
    for zoom in zooms:
        new_w = int(round(w * zoom))
        new_h = int(round(h * zoom))
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        crop = resized.crop((left, top, left + w, top + h))
        out += as_array(crop)
    out /= float(len(zooms) + 1)
    return to_image(out)


def snow(image, severity, rng):
    x = as_array(image)
    h, w = x.shape[:2]
    density = [0.06, 0.09, 0.13, 0.18, 0.24][severity - 1]
    snow_layer = rng.random((h, w)) < density
    snow_layer = Image.fromarray((snow_layer.astype(np.uint8) * 255), mode="L")
    snow_layer = snow_layer.filter(ImageFilter.GaussianBlur(radius=[0.6, 0.8, 1.0, 1.2, 1.5][severity - 1]))
    s = np.asarray(snow_layer).astype(np.float32) / 255.0
    x = x * (1.0 - 0.12 * severity) + 0.12 * severity
    x = np.maximum(x, s[..., None])
    return to_image(x)


def frost(image, severity, rng):
    x = as_array(image)
    h, w = x.shape[:2]
    noise = rng.normal(0.5, 0.25, (h, w)).clip(0, 1)
    frost_layer = Image.fromarray((noise * 255).astype(np.uint8), mode="L")
    frost_layer = frost_layer.filter(ImageFilter.GaussianBlur(radius=2 + severity))
    f = np.asarray(frost_layer).astype(np.float32) / 255.0
    frost_color = np.array([0.85, 0.92, 1.0], dtype=np.float32)
    alpha = [0.18, 0.25, 0.33, 0.42, 0.52][severity - 1] * f[..., None]
    x = x * (1.0 - alpha) + frost_color * alpha
    return to_image(x)


def fog(image, severity, rng):
    x = as_array(image)
    h, w = x.shape[:2]
    noise = rng.random((h, w))
    fog_layer = Image.fromarray((noise * 255).astype(np.uint8), mode="L")
    fog_layer = fog_layer.filter(ImageFilter.GaussianBlur(radius=12 + severity * 5))
    f = np.asarray(fog_layer).astype(np.float32) / 255.0
    alpha = [0.18, 0.26, 0.35, 0.45, 0.56][severity - 1]
    x = x * (1.0 - alpha * f[..., None]) + alpha * f[..., None]
    return to_image(x)


def brightness(image, severity, rng):
    factors = [1.15, 1.30, 1.45, 1.60, 1.75]
    return ImageEnhance.Brightness(image).enhance(factors[severity - 1])


def contrast(image, severity, rng):
    factors = [0.85, 0.70, 0.55, 0.40, 0.25]
    return ImageEnhance.Contrast(image).enhance(factors[severity - 1])


def elastic_transform(image, severity, rng):
    x = np.asarray(image.convert("RGB"))
    h, w = x.shape[:2]
    amp = [2, 4, 6, 8, 10][severity - 1]
    period = max(20, min(h, w) // 8)
    out = np.empty_like(x)
    for y in range(h):
        dx = int(round(amp * math.sin(2.0 * math.pi * y / period)))
        out[y] = np.roll(x[y], dx, axis=0)
    final = np.empty_like(out)
    for col in range(w):
        dy = int(round(amp * math.sin(2.0 * math.pi * col / period)))
        final[:, col] = np.roll(out[:, col], dy, axis=0)
    return Image.fromarray(final, mode="RGB")


def jpeg_compression(image, severity, rng):
    qualities = [75, 55, 35, 20, 10]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=qualities[severity - 1])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def pixelate(image, severity, rng):
    scales = [0.75, 0.60, 0.50, 0.40, 0.30]
    w, h = image.size
    scale = scales[severity - 1]
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


CORRUPTION_FNS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "jpeg_compression": jpeg_compression,
    "pixelate": pixelate,
}


def patch_imagecorruptions_compat():
    if not hasattr(np, "float_"):
        np.float_ = np.float64

    if imagecorruptions_corruptions is None:
        return
    gaussian = getattr(imagecorruptions_corruptions, "gaussian", None)
    if gaussian is None or getattr(gaussian, "_fasterrcnn_compat", False):
        return

    params = inspect.signature(gaussian).parameters
    supports_channel_axis = "channel_axis" in params
    supports_multichannel = "multichannel" in params

    def gaussian_compat(
        image,
        sigma=1,
        output=None,
        mode="nearest",
        cval=0,
        preserve_range=False,
        truncate=4.0,
        *,
        multichannel=None,
        channel_axis=None,
        **kwargs,
    ):
        call_kwargs = {}
        if "sigma" in params:
            call_kwargs["sigma"] = sigma
        if "output" in params and output is not None:
            call_kwargs["output"] = output
        if "out" in params and output is not None:
            call_kwargs["out"] = output
        if "mode" in params:
            call_kwargs["mode"] = mode
        if "cval" in params:
            call_kwargs["cval"] = cval
        if "preserve_range" in params:
            call_kwargs["preserve_range"] = preserve_range
        if "truncate" in params:
            call_kwargs["truncate"] = truncate
        if supports_channel_axis:
            if multichannel is not None and channel_axis is None:
                channel_axis = -1 if multichannel else None
            call_kwargs["channel_axis"] = channel_axis
        elif supports_multichannel:
            if multichannel is None:
                multichannel = channel_axis is not None
            call_kwargs["multichannel"] = multichannel
        for key, value in kwargs.items():
            if key in params:
                call_kwargs[key] = value
        return gaussian(image, **call_kwargs)

    gaussian_compat._fasterrcnn_compat = True
    imagecorruptions_corruptions.gaussian = gaussian_compat


def apply_corruption(image, corruption, severity, backend):
    if backend == "imagecorruptions":
        if imagecorruptions_corrupt is None:
            raise RuntimeError(
                "imagecorruptions is required for a benchmark comparable with "
                "MMDetection's type='Corrupt' transform. Install imagecorruptions "
                "in this environment or run with --corruption-backend local for "
                "the older approximate implementation."
            )
        patch_imagecorruptions_compat()
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        corrupted = imagecorruptions_corrupt(
            array,
            corruption_name=corruption,
            severity=int(severity),
        )
        return Image.fromarray(corrupted.astype(np.uint8), mode="RGB")

    if backend == "local":
        rng = np.random.default_rng(stable_seed(image.filename if hasattr(image, "filename") else "", corruption, severity))
        return CORRUPTION_FNS[corruption](image, int(severity), rng)

    raise ValueError("Unknown corruption backend: {}".format(backend))


class CorruptedCocoDataset(Dataset):
    def __init__(self, annotation_json, corruption="Clean", severity=0, corruption_backend="imagecorruptions"):
        self.annotation_json = Path(annotation_json)
        self.corruption = corruption
        self.severity = int(severity)
        self.corruption_backend = corruption_backend
        data = json.loads(self.annotation_json.read_text(encoding="utf-8"))
        self.images = data["images"]
        self.categories = data["categories"]
        self.annotations_by_image = {image["id"]: [] for image in self.images}
        for ann in data["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image = Image.open(info["file_name"]).convert("RGB")
        width, height = image.size
        if self.corruption != "Clean" and self.severity > 0:
            image.filename = info["file_name"]
            image = apply_corruption(image, self.corruption, self.severity, self.corruption_backend)

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

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(info["id"])]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return F.to_tensor(image), target


def discover_weight_files(project, run_glob):
    root = Path(project)
    weight_files = []
    for pattern in str(run_glob).split(","):
        pattern = pattern.strip()
        if pattern:
            weight_files.extend(root.glob("{}/last.pth".format(pattern)))
    return sorted(set(weight_files))


def run_name(weight_path):
    return Path(weight_path).parent.name


def metric_columns():
    cols = ["map50_95", "map50"]
    for class_name in CLASS_NAMES:
        cols.append("AP50_95_{}".format(class_name))
    for class_name in CLASS_NAMES:
        cols.append("AP50_{}".format(class_name))
    return cols


def evaluate_condition(args, model, dataset, device, metric_cls):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    metric = metric_cls(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        extended_summary=True,
        max_detection_thresholds=[1, 10, args.metric_max_detections],
    )
    if hasattr(metric, "warn_on_many_detections"):
        metric.warn_on_many_detections = False

    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(images)
            preds = []
            metric_targets = []
            for output in outputs:
                scores = output["scores"].detach().cpu()
                keep = scores >= args.score_threshold
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
    ap_by_class = per_class_values(result, "map_per_class")
    ap50_by_class = per_class_values(result, "map_50_per_class")
    if not ap50_by_class:
        ap50_by_class = ap50_from_extended_summary(result)

    row = {
        "map50_95": metric_value(result, "map"),
        "map50": metric_value(result, "map_50"),
    }
    for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
        row["AP50_95_{}".format(class_name)] = ap_by_class.get(class_id, "")
    for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
        row["AP50_{}".format(class_name)] = ap50_by_class.get(class_id, "")
    return row


def load_completed(raw_csv):
    completed = set()
    if not raw_csv.exists():
        return completed
    with raw_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["run"], row["corruption"], int(row["severity"])))
    return completed


def append_raw_row(raw_csv, row):
    exists = raw_csv.exists()
    fieldnames = ["run", "corruption", "severity"] + metric_columns()
    with raw_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_raw(raw_csv):
    rows = []
    if not raw_csv.exists():
        return rows
    with raw_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def float_or_none(value):
    if value == "" or value is None:
        return None
    return float(value)


def aggregate_value(rows, run, corruption, metric):
    values = []
    for row in rows:
        if row["run"] != run:
            continue
        if row["corruption"] != corruption:
            continue
        value = float_or_none(row.get(metric))
        if value is not None:
            values.append(value)
    if not values:
        return ""
    return sum(values) / float(len(values))


def write_summary_tables(out_dir, raw_csv):
    rows = read_raw(raw_csv)
    runs = sorted(set(row["run"] for row in rows))
    summary_dir = out_dir / "tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    table_paths = []
    for metric in metric_columns():
        out_path = summary_dir / "{}.csv".format(metric.replace(".", ""))
        table_paths.append(out_path)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for run in runs:
                clean = aggregate_value(rows, run, "Clean", metric)
                row = {"run": run, "Clean": clean}
                corruption_values = []
                for corruption in CORRUPTIONS:
                    display = DISPLAY_NAMES[corruption]
                    value = aggregate_value(rows, run, corruption, metric)
                    row[display] = value
                    if value != "":
                        corruption_values.append(float(value))
                if corruption_values:
                    mpc = sum(corruption_values) / float(len(corruption_values))
                    row["mPC"] = mpc
                    if clean != "" and float(clean) > 0:
                        row["rPC"] = 100.0 * mpc / float(clean)
                    else:
                        row["rPC"] = ""
                else:
                    row["mPC"] = ""
                    row["rPC"] = ""
                writer.writerow(row)
    return table_paths


def write_manifest(out_dir, args, weight_files):
    dependency_versions = {}
    for package in ["imagecorruptions", "scikit-image", "torch", "torchvision", "torchmetrics"]:
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    payload = {
        "data": str(Path(args.data).resolve()),
        "project": str(Path(args.project).resolve()),
        "run_glob": args.run_glob,
        "num_models": len(weight_files),
        "models": [run_name(path) for path in weight_files],
        "corruptions": CORRUPTIONS,
        "corruption_backend": args.corruption_backend,
        "dependency_versions": dependency_versions,
        "severities": [1, 2, 3, 4, 5],
        "columns": SUMMARY_COLUMNS,
        "metric_note": "Clean is evaluated once. Each corruption column is the mean over severities 1-5. mPC is the mean over the 15 corruption columns. rPC is 100*mPC/Clean.",
    }
    (out_dir / "benchmark_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=project_root / "data_cityscapes_val" / "cityscapes_val_coco.json")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=project_root / "runs" / "train" / "cityscapes_15fixed_corruption_benchmark")
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--metric-max-detections", type=int, default=100)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--weights-type", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument(
        "--corruption-backend",
        choices=["imagecorruptions", "local"],
        default="imagecorruptions",
        help="Use imagecorruptions for parity with MMDetection type='Corrupt'; local is the older approximate implementation.",
    )
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("Evaluation COCO json not found: {}".format(args.data))

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise SystemExit("torchmetrics detection mAP backend unavailable: {}".format(exc))

    if args.corruption_backend == "imagecorruptions" and imagecorruptions_corrupt is None:
        raise SystemExit(
            "imagecorruptions is not installed in this Python environment. "
            "Install it to match test_robustness.py/MMDetection Corrupt, or run "
            "with --corruption-backend local only for approximate non-comparable tests."
        )

    weight_files = discover_weight_files(args.project, args.run_glob)
    if args.limit_runs > 0:
        weight_files = weight_files[: args.limit_runs]
    if not weight_files:
        raise SystemExit("No last.pth files found under {} with run glob {}".format(args.project, args.run_glob))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "raw_results.csv"
    completed = load_completed(raw_csv)
    device = torch.device(args.device)

    clean_dataset = CorruptedCocoDataset(
        args.data,
        corruption="Clean",
        severity=0,
        corruption_backend=args.corruption_backend,
    )
    num_classes = len(clean_dataset.categories) + 1

    write_manifest(args.out_dir, args, weight_files)

    conditions = [("Clean", 0)]
    if not args.clean_only:
        for corruption in CORRUPTIONS:
            for severity in [1, 2, 3, 4, 5]:
                conditions.append((corruption, severity))

    for weight_idx, weight_path in enumerate(weight_files, 1):
        run = run_name(weight_path)
        print("[{}/{}] Loading {}".format(weight_idx, len(weight_files), run))
        model = build_model(
            num_classes=num_classes,
            trainable_layers=args.trainable_layers,
            min_size=args.min_size,
            max_size=args.max_size,
            pretrained_backbone=False,
        )
        load_checkpoint(weight_path, model, device=device, weights=args.weights_type)
        model.roi_heads.score_thresh = args.score_threshold
        model.roi_heads.detections_per_img = args.detections_per_img
        model.to(device)
        model.eval()

        for condition_idx, (corruption, severity) in enumerate(conditions, 1):
            key = (run, corruption, severity)
            if key in completed:
                print("  [{}/{}] Skip {} severity {}".format(condition_idx, len(conditions), corruption, severity))
                continue
            print("  [{}/{}] Evaluating {} severity {}".format(condition_idx, len(conditions), corruption, severity))
            dataset = CorruptedCocoDataset(
                args.data,
                corruption=corruption,
                severity=severity,
                corruption_backend=args.corruption_backend,
            )
            metrics = evaluate_condition(args, model, dataset, device, MeanAveragePrecision)
            row = {"run": run, "corruption": corruption, "severity": severity}
            row.update(metrics)
            append_raw_row(raw_csv, row)
            completed.add(key)
            write_summary_tables(args.out_dir, raw_csv)
            print(
                "    map50_95={} map50={}".format(
                    row["map50_95"], row["map50"]
                )
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tables = write_summary_tables(args.out_dir, raw_csv)
    print("Saved raw results: {}".format(raw_csv))
    print("Saved summary tables:")
    for path in tables:
        print("  {}".format(path))


if __name__ == "__main__":
    main()

"""Evaluate discovered Faster R-CNN checkpoints on a prepared BDD100K split.

All dataset, checkpoint, and CSV output locations are supplied by the CLI.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_fasterrcnn_resnet101 import (
    CocoStyleDetectionDataset,
    build_model,
    collate_fn,
    load_checkpoint,
)


CLASS_NAMES = ["Bicycle", "Bus", "Car", "Motor", "Psn.", "Rider", "Truck"]
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7]


def to_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
    if value is None:
        return ""
    return float(value)


def metric_value(result, key):
    if key not in result:
        return ""
    value = result[key]
    if hasattr(value, "detach") and value.numel() == 1:
        value = float(value.detach().cpu())
        return "" if value < 0 else value
    return ""


def per_class_values(result, key):
    if key not in result or "classes" not in result:
        return {}
    values = result[key].detach().cpu()
    classes = result["classes"].detach().cpu()
    out = {}
    for cls, value in zip(classes.tolist(), values.tolist()):
        if value >= 0:
            out[int(cls)] = float(value)
    return out


def ap50_from_extended_summary(result):
    if "precision" not in result or "classes" not in result:
        return {}

    precision = result["precision"].detach().cpu()
    classes = result["classes"].detach().cpu()

    # torchmetrics/COCO precision shape is [IoU, Recall, Class, Area, MaxDets].
    # Index 0 is IoU=0.50, area index 0 is "all", and -1 uses the metric maxDets.
    if precision.ndim != 5:
        return {}

    out = {}
    for class_idx, cls in enumerate(classes.tolist()):
        values = precision[0, :, class_idx, 0, -1]
        values = values[values >= 0]
        if values.numel() > 0:
            out[int(cls)] = float(values.mean().item())
    return out


def discover_weight_files(project, run_glob):
    train_dir = Path(project)
    weight_files = []
    for pattern in str(run_glob).split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        weight_files.extend(train_dir.glob("{}/last.pth".format(pattern)))
    return sorted(set(weight_files))


def run_name_from_weight(weight_path):
    return weight_path.parent.name


def read_run_config(weight_path):
    config_path = weight_path.parent / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def evaluate_one(args, weight_path, dataset, device, metric_cls):
    num_classes = len(dataset.categories) + 1
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
    pred_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)
            preds = []
            metric_targets = []
            for output in outputs:
                scores = output["scores"].detach().cpu()
                keep = scores >= args.score_threshold
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
    ap_by_class = per_class_values(result, "map_per_class")
    ap50_by_class = per_class_values(result, "map_50_per_class")
    if not ap50_by_class:
        ap50_by_class = ap50_from_extended_summary(result)

    row = {
        "run": run_name_from_weight(weight_path),
        "weights": str(weight_path),
        "images": len(dataset),
        "predictions": pred_count,
        "avg_predictions_per_image": pred_count / float(max(1, len(dataset))),
        "map50_95": metric_value(result, "map"),
        "map50": metric_value(result, "map_50"),
        "map75": metric_value(result, "map_75"),
    }
    for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
        row["AP_{}".format(class_name)] = ap_by_class.get(class_id, "")
        row["AP50_{}".format(class_name)] = ap50_by_class.get(class_id, "")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=project_root / "data_bdd100k" / "bdd100k_val_coco.json")
    parser.add_argument("--project", type=Path, default=project_root / "runs" / "train")
    parser.add_argument("--out", type=Path, default=project_root / "runs" / "bdd100k_val_all_models.csv")
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
    parser.add_argument("--run-glob", default="*")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("Evaluation COCO json not found: {}".format(args.data))

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise SystemExit("torchmetrics detection mAP backend unavailable: {}".format(exc))

    weight_files = discover_weight_files(args.project, args.run_glob)
    if not weight_files:
        raise SystemExit("No last.pth files found under {} with run glob {}".format(args.project, args.run_glob))

    dataset = CocoStyleDetectionDataset(args.data, train=False)
    device = torch.device(args.device)

    fieldnames = [
        "run",
        "weights",
        "images",
        "predictions",
        "avg_predictions_per_image",
        "map50_95",
        "map50",
        "map75",
    ]
    for class_name in CLASS_NAMES:
        fieldnames.append("AP_{}".format(class_name))
    for class_name in CLASS_NAMES:
        fieldnames.append("AP50_{}".format(class_name))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, weight_path in enumerate(weight_files, 1):
            print("[{}/{}] Evaluating {}".format(idx, len(weight_files), weight_path.parent.name))
            row = evaluate_one(args, weight_path, dataset, device, MeanAveragePrecision)
            writer.writerow(row)
            f.flush()
            print(
                "  map50_95={} map50={} AP_Car={} AP_Psn={}".format(
                    row["map50_95"], row["map50"], row["AP_Car"], row["AP_Psn."]
                )
            )

    print("Saved evaluation summary: {}".format(args.out))


if __name__ == "__main__":
    main()

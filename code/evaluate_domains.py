"""Evaluate one Faster R-CNN checkpoint across prepared target domains.

Domain definitions and all filesystem locations are passed explicitly by CLI.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import torch
from train_fasterrcnn_resnet101 import (
    CocoStyleDetectionDataset,
    build_model,
    collate_fn,
    load_checkpoint,
)
from torch.utils.data import DataLoader


def to_float(value):
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_root
        / "runs"
        / "train"
        / "frcnn_r101_sgd"
        / "last.pth",
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "domain_eval")
    parser.add_argument("--out", type=Path, default=project_root / "runs" / "domain_eval_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--weights-type", choices=["auto", "model", "ema"], default="auto")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit("Weights not found: {}".format(args.weights))

    json_files = sorted(args.data_dir.glob("*_coco.json"))
    if not json_files:
        raise SystemExit("No domain eval json files found in {}".format(args.data_dir))

    first = json.loads(json_files[0].read_text(encoding="utf-8"))
    num_classes = len(first["categories"]) + 1
    device = torch.device(args.device)

    model = build_model(
        num_classes, args.trainable_layers, args.min_size, args.max_size,
        pretrained_backbone=False,
    )
    load_checkpoint(args.weights, model, device=device, weights=args.weights_type)
    model.to(device)
    model.eval()
    print("Loaded weights: {} ({})".format(args.weights, args.weights_type))

    metric_cls = None
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric_cls = MeanAveragePrecision
    except Exception as exc:
        print("torchmetrics mAP backend unavailable: {}".format(exc))
        print("Falling back to prediction-count summary.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style",
                "images",
                "predictions",
                "avg_predictions_per_image",
                "map50_95",
                "map50",
                "map75",
            ],
        )
        writer.writeheader()
        for json_path in json_files:
            style = json_path.name.replace("_coco.json", "")
            dataset = CocoStyleDetectionDataset(json_path, train=False)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=collate_fn,
            )
            pred_count = 0
            metric = metric_cls(box_format="xyxy", iou_type="bbox") if metric_cls else None
            with torch.no_grad():
                for images, targets in loader:
                    images = [img.to(device, non_blocking=True) for img in images]
                    outputs = model(images)
                    metric_preds = []
                    metric_targets = []
                    for output in outputs:
                        scores = output["scores"].detach().cpu()
                        keep = scores >= args.score_threshold
                        pred_count += int(keep.sum().item())
                        if metric is not None:
                            metric_preds.append(
                                {
                                    "boxes": output["boxes"].detach().cpu()[keep],
                                    "scores": scores[keep],
                                    "labels": output["labels"].detach().cpu()[keep],
                                }
                            )
                    if metric is not None:
                        for target in targets:
                            metric_targets.append(
                                {
                                    "boxes": target["boxes"].detach().cpu(),
                                    "labels": target["labels"].detach().cpu(),
                                }
                            )
                        metric.update(metric_preds, metric_targets)

            map_all = ""
            map50 = ""
            map75 = ""
            if metric is not None:
                result = metric.compute()
                map_all = to_float(result["map"])
                map50 = to_float(result["map_50"])
                map75 = to_float(result["map_75"])

            writer.writerow(
                {
                    "style": style,
                    "images": len(dataset),
                    "predictions": pred_count,
                    "avg_predictions_per_image": pred_count / float(max(1, len(dataset))),
                    "map50_95": map_all,
                    "map50": map50,
                    "map75": map75,
                }
            )
            print(
                "{} images={} predictions={} map50_95={} map50={}".format(
                    style, len(dataset), pred_count, map_all, map50
                )
            )

    print("Saved domain evaluation summary: {}".format(args.out))


if __name__ == "__main__":
    main()

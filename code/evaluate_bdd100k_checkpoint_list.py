#!/usr/bin/env python3
"""Evaluate a CSV-listed set of Faster R-CNN checkpoints on BDD100K.

The manifest fixes checkpoint identity and supports reproducible shard selection.
"""

from __future__ import print_function

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_bdd100k_all import (  # noqa: E402
    CLASS_NAMES,
    CocoStyleDetectionDataset,
    evaluate_one,
)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
        "--resume",
        action="store_true",
        help="Keep completed rows in --out and skip matching run/checkpoint pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise SystemExit("Evaluation COCO json not found: {}".format(args.data))
    if not args.manifest.exists():
        raise SystemExit("Checkpoint manifest not found: {}".format(args.manifest))

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise SystemExit("torchmetrics detection mAP backend unavailable: {}".format(exc))

    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    if not manifest_rows:
        raise SystemExit("Empty checkpoint manifest: {}".format(args.manifest))

    dataset = CocoStyleDetectionDataset(args.data, train=False)
    device = torch.device(args.device)
    eval_args = SimpleNamespace(
        trainable_layers=args.trainable_layers,
        min_size=args.min_size,
        max_size=args.max_size,
        weights_type=args.weights_type,
        score_threshold=args.score_threshold,
        detections_per_img=args.detections_per_img,
        batch_size=args.batch_size,
        workers=args.workers,
        metric_max_detections=args.metric_max_detections,
    )

    metric_fields = [
        "images",
        "predictions",
        "avg_predictions_per_image",
        "map50_95",
        "map50",
        "map75",
    ]
    for class_name in CLASS_NAMES:
        metric_fields.append("AP_{}".format(class_name))
    for class_name in CLASS_NAMES:
        metric_fields.append("AP50_{}".format(class_name))

    out_fields = [
        "exp",
        "description",
        "run",
        "checkpoint_tag",
        "checkpoint",
        "consistency_mode",
        "global_weight",
        "target_weight",
        "class_weight",
        "box_weight",
        "scene_repeat_threshold",
        "target_consistency_max_objects",
    ] + metric_fields

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    write_header = True
    open_mode = "w"
    if args.resume and args.out.exists() and args.out.stat().st_size > 0:
        with args.out.open(newline="", encoding="utf-8") as existing_file:
            existing_rows = list(csv.DictReader(existing_file))
        completed = {
            (row.get("run", ""), row.get("checkpoint", ""))
            for row in existing_rows
        }
        write_header = False
        open_mode = "a"

    with args.out.open(open_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            writer.writeheader()
        for idx, manifest_row in enumerate(manifest_rows, 1):
            checkpoint = Path(manifest_row["checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            completed_key = (manifest_row["run"], str(checkpoint))
            if completed_key in completed:
                print(
                    "[{}/{}] Skip completed {} {}".format(
                        idx,
                        len(manifest_rows),
                        manifest_row["run"],
                        manifest_row["checkpoint_tag"],
                    )
                )
                continue
            print(
                "[{}/{}] Evaluating {} {}".format(
                    idx,
                    len(manifest_rows),
                    manifest_row["run"],
                    manifest_row["checkpoint_tag"],
                ))
            metric_row = evaluate_one(eval_args, checkpoint, dataset, device, MeanAveragePrecision)
            out = {
                "exp": manifest_row.get("exp", ""),
                "description": manifest_row.get("description", ""),
                "run": manifest_row["run"],
                "checkpoint_tag": manifest_row["checkpoint_tag"],
                "checkpoint": str(checkpoint),
                "consistency_mode": manifest_row.get("consistency_mode", ""),
                "global_weight": manifest_row.get("global_weight", ""),
                "target_weight": manifest_row.get("target_weight", ""),
                "class_weight": manifest_row.get("class_weight", ""),
                "box_weight": manifest_row.get("box_weight", ""),
                "scene_repeat_threshold": manifest_row.get(
                    "scene_repeat_threshold", ""),
                "target_consistency_max_objects": manifest_row.get(
                    "target_consistency_max_objects", ""),
            }
            out.update({key: metric_row.get(key, "") for key in metric_fields})
            writer.writerow(out)
            f.flush()
            completed.add(completed_key)
            print("  map50_95={} map50={}".format(out["map50_95"], out["map50"]))

    print("Saved checkpoint evaluation: {}".format(args.out))


if __name__ == "__main__":
    main()

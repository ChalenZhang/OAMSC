#!/usr/bin/env python3
"""Evaluate YOLOv8 checkpoints enumerated by a CSV manifest.

The manifest specifies checkpoint tags, subsets, and output locations.
"""

from __future__ import print_function

import argparse
import csv
from pathlib import Path


CLASS_NAMES = ["Bicycle", "Bus", "Car", "Motor", "Psn.", "Rider", "Truck"]


def metric_value(value):
    try:
        return float(value)
    except Exception:
        return ""


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Empty checkpoint manifest: {}".format(path))
    return rows


def completed_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("run", ""), row.get("checkpoint", ""))
            for row in csv.DictReader(handle)
        }


def class_metrics(box):
    ap5095 = {}
    ap50 = {}
    maps = getattr(box, "maps", None)
    if maps is not None:
        for class_index, value in enumerate(maps):
            ap5095[class_index] = metric_value(value)

    all_ap = getattr(box, "all_ap", None)
    ap_class_index = getattr(box, "ap_class_index", None)
    if all_ap is not None and ap_class_index is not None:
        for row_index, class_index in enumerate(ap_class_index):
            class_index = int(class_index)
            if row_index < len(all_ap) and len(all_ap[row_index]) > 0:
                ap50[class_index] = metric_value(all_ap[row_index][0])
    return ap5095, ap50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("YOLO dataset yaml not found: {}".format(args.data))
    if not args.manifest.exists():
        raise SystemExit("Checkpoint manifest not found: {}".format(args.manifest))

    from ultralytics import YOLO

    manifest_rows = read_manifest(args.manifest)
    fields = [
        "exp",
        "description",
        "architecture",
        "source",
        "style_count",
        "run",
        "checkpoint_tag",
        "checkpoint",
        "map50_95",
        "map50",
        "map75",
    ]
    fields.extend("AP_{}".format(name) for name in CLASS_NAMES)
    fields.extend("AP50_{}".format(name) for name in CLASS_NAMES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_rows(args.out) if args.resume else set()
    mode = "a" if args.resume and args.out.exists() and args.out.stat().st_size else "w"
    validation_root = args.out.parent / "ultralytics_validation"

    with args.out.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for index, manifest_row in enumerate(manifest_rows, 1):
            checkpoint = Path(manifest_row["checkpoint"])
            key = (manifest_row["run"], str(checkpoint))
            if key in completed:
                print(
                    "[{}/{}] Skip completed {}".format(
                        index, len(manifest_rows), manifest_row["run"]
                    )
                )
                continue
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            print(
                "[{}/{}] Evaluating {}".format(
                    index, len(manifest_rows), manifest_row["run"]
                )
            )
            model = YOLO(str(checkpoint))
            metrics = model.val(
                data=str(args.data),
                split="val",
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                project=str(validation_root),
                name=manifest_row["run"],
                exist_ok=True,
                plots=False,
                save_json=False,
                verbose=False,
            )
            box = metrics.box
            ap5095, ap50 = class_metrics(box)
            row = {
                "exp": manifest_row.get("exp", ""),
                "description": manifest_row.get("description", ""),
                "architecture": manifest_row.get("architecture", "yolov8x"),
                "source": manifest_row.get("source", ""),
                "style_count": manifest_row.get("style_count", ""),
                "run": manifest_row["run"],
                "checkpoint_tag": manifest_row.get("checkpoint_tag", "last"),
                "checkpoint": str(checkpoint),
                "map50_95": metric_value(getattr(box, "map", "")),
                "map50": metric_value(getattr(box, "map50", "")),
                "map75": metric_value(getattr(box, "map75", "")),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row["AP_{}".format(class_name)] = ap5095.get(class_index, "")
                row["AP50_{}".format(class_name)] = ap50.get(class_index, "")
            writer.writerow(row)
            handle.flush()
            completed.add(key)
            print(
                "  map50_95={} map50={}".format(row["map50_95"], row["map50"])
            )

    print("Saved YOLO checkpoint evaluation: {}".format(args.out))


if __name__ == "__main__":
    main()

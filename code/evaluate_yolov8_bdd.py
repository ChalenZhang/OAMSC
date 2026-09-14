"""Evaluate a YOLOv8 checkpoint on the configured BDD100K validation split.

The resulting CSV uses the seven-class label order documented in the README.
"""

from __future__ import print_function

import argparse
import csv
from pathlib import Path


CLASS_NAMES = ["Bicycle", "Bus", "Car", "Motor", "Psn.", "Rider", "Truck"]


def discover_weights(project, run_glob):
    weights = []
    for pattern in str(run_glob).split(","):
        pattern = pattern.strip()
        if not pattern:
            continue
        root = Path(project)
        weights.extend(root.glob("{}/weights/best.pt".format(pattern)))
        weights.extend(root.glob("{}/weights/last.pt".format(pattern)))
    best = {}
    for path in sorted(set(weights)):
        run = path.parents[1].name
        if run not in best or path.name == "best.pt":
            best[run] = path
    return [best[key] for key in sorted(best)]


def metric_value(value):
    try:
        return float(value)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = discover_weights(args.project, args.run_glob)
    if not weights:
        raise SystemExit("No YOLO weights found under {} with glob {}".format(args.project, args.run_glob))

    fieldnames = ["run", "weights", "map50_95", "map50", "map75"]
    for name in CLASS_NAMES:
        fieldnames.append("AP_{}".format(name))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, weight in enumerate(weights, 1):
            run = weight.parents[1].name
            print("[{}/{}] Evaluating {}".format(idx, len(weights), run))
            model = YOLO(str(weight))
            metrics = model.val(
                data=str(args.data),
                split="val",
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )
            box = metrics.box
            row = {
                "run": run,
                "weights": str(weight),
                "map50_95": metric_value(getattr(box, "map", "")),
                "map50": metric_value(getattr(box, "map50", "")),
                "map75": metric_value(getattr(box, "map75", "")),
            }
            maps = getattr(box, "maps", None)
            if maps is not None:
                for class_idx, name in enumerate(CLASS_NAMES):
                    if class_idx < len(maps):
                        row["AP_{}".format(name)] = metric_value(maps[class_idx])
            writer.writerow(row)
            f.flush()
            print("  map50_95={} map50={}".format(row["map50_95"], row["map50"]))
    print("Saved YOLO evaluation summary: {}".format(args.out))


if __name__ == "__main__":
    main()

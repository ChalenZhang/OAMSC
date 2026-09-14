"""Convert a filtered BDD100K split into seven-class COCO detection records.

CLI filters select weather and time-of-day subsets; CSV/JSON summaries expose
retained, skipped, and class-level counts for review.
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import IMAGE_EXTS, image_size, norm_path


CATEGORIES = [
    {"id": 1, "name": "Bicycle", "source_bdd_category": "bike"},
    {"id": 2, "name": "Bus", "source_bdd_category": "bus"},
    {"id": 3, "name": "Car", "source_bdd_category": "car"},
    {"id": 4, "name": "Motor", "source_bdd_category": "motor"},
    {"id": 5, "name": "Psn.", "source_bdd_category": "person"},
    {"id": 6, "name": "Rider", "source_bdd_category": "rider"},
    {"id": 7, "name": "Truck", "source_bdd_category": "truck"},
]

BDD_TO_CATEGORY_ID = {
    "bike": 1,
    "bus": 2,
    "car": 3,
    "motor": 4,
    "person": 5,
    "rider": 6,
    "truck": 7,
}


def collect_images(image_dir):
    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def collect_labels(label_dir):
    return {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    }


def load_bdd_objects(label_path):
    data = json.loads(label_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        return []
    return frames[0].get("objects", [])


def load_bdd_label(label_path):
    return json.loads(label_path.read_text(encoding="utf-8"))


def normalize_requested_token(value):
    value = str(value).strip()
    lowered = value.lower()
    if lowered == "day":
        return "daytime"
    if lowered in ("dark", "nighttime"):
        return "night"
    return value


def requested_tokens(requested):
    if requested is None:
        return []
    requested = str(requested).strip()
    if requested.lower() in ("", "all", "*", "any"):
        return []
    return [
        normalize_requested_token(part)
        for part in requested.replace(";", ",").split(",")
        if part.strip()
    ]


def selected_attr(value, requested):
    if requested is None:
        return True
    requested = str(requested).strip()
    if requested.lower() in ("", "all", "*", "any"):
        return True
    if requested.lower() in ("not_clear", "non_clear", "except_clear", "all_except_clear"):
        return str(value) != "clear"
    tokens = requested_tokens(requested)
    return str(value) in set(tokens)


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "BDD100K_val")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_bdd100k")
    parser.add_argument("--name", default="bdd100k_val_daytime_clear")
    parser.add_argument("--timeofday", default="daytime", help="BDD timeofday filter, or all")
    parser.add_argument("--weather", default="clear", help="BDD weather filter, or all")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    image_dir = dataset_root / "images"
    label_dir = dataset_root / "labels"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_by_stem = collect_images(image_dir)
    labels_by_stem = collect_labels(label_dir)
    missing_labels = sorted(set(images_by_stem) - set(labels_by_stem))
    orphan_labels = sorted(set(labels_by_stem) - set(images_by_stem))

    coco_images = []
    annotations = []
    manifest_rows = []
    category_counts = Counter()
    ignored_category_counts = Counter()
    invalid_box_counts = Counter()
    timeofday_counts = Counter()
    weather_counts = Counter()
    selected_timeofday_counts = Counter()
    selected_weather_counts = Counter()
    bad_images = []
    bad_labels = []
    filtered_out = 0
    ann_id = 1
    image_id = 1

    for stem in sorted(images_by_stem):
        image_path = images_by_stem[stem]
        label_path = labels_by_stem.get(stem)
        if label_path is None:
            continue

        try:
            label_data = load_bdd_label(label_path)
        except Exception as exc:
            bad_labels.append(
                {
                    "scene": stem,
                    "label_path": norm_path(label_path),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            continue

        attrs = label_data.get("attributes", {})
        timeofday = attrs.get("timeofday")
        weather = attrs.get("weather")
        timeofday_counts[timeofday] += 1
        weather_counts[weather] += 1
        if not selected_attr(timeofday, args.timeofday) or not selected_attr(weather, args.weather):
            filtered_out += 1
            continue
        selected_timeofday_counts[timeofday] += 1
        selected_weather_counts[weather] += 1

        try:
            width, height = image_size(image_path)
        except Exception as exc:
            bad_images.append(
                {
                    "scene": stem,
                    "image_path": norm_path(image_path),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            continue

        coco_images.append(
            {
                "id": image_id,
                "file_name": norm_path(image_path),
                "width": width,
                "height": height,
                "scene": stem,
                "style": "BDD100K_val",
                "timeofday": timeofday,
                "weather": weather,
                "label_file": norm_path(label_path),
            }
        )

        kept = 0
        ignored = 0
        invalid = 0
        frames = label_data.get("frames", [])
        objects = frames[0].get("objects", []) if frames else []
        for obj in objects:
            source_category = obj.get("category")
            category_id = BDD_TO_CATEGORY_ID.get(source_category)
            if category_id is None:
                if isinstance(obj.get("box2d"), dict):
                    ignored_category_counts[source_category] += 1
                    ignored += 1
                continue

            box = obj.get("box2d")
            if not isinstance(box, dict):
                invalid_box_counts[source_category] += 1
                invalid += 1
                continue

            try:
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
            except Exception:
                invalid_box_counts[source_category] += 1
                invalid += 1
                continue

            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 1.0 or box_h <= 1.0:
                invalid_box_counts[source_category] += 1
                invalid += 1
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "source_bdd_category": source_category,
                }
            )
            ann_id += 1
            kept += 1
            category_counts[source_category] += 1

        manifest_rows.append(
            {
                "scene": stem,
                "image_path": norm_path(image_path),
                "label_path": norm_path(label_path),
                "width": width,
                "height": height,
                "timeofday": timeofday,
                "weather": weather,
                "kept_annotations": kept,
                "ignored_annotations": ignored,
                "invalid_annotations": invalid,
            }
        )
        image_id += 1

    output_json = output_dir / "{}_coco.json".format(args.name)
    data = {
        "info": {
            "description": "BDD100K val converted to COCO-style boxes for Faster R-CNN evaluation",
            "label_note": "Only bike, bus, car, motor, person, rider, truck are kept and mapped to the 7 training classes.",
            "timeofday_filter": args.timeofday,
            "weather_filter": args.weather,
        },
        "images": coco_images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_csv = output_dir / "{}_manifest.csv".format(args.name)
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "image_path",
                "label_path",
                "width",
                "height",
                "timeofday",
                "weather",
                "kept_annotations",
                "ignored_annotations",
                "invalid_annotations",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    summary = {
        "dataset_root": norm_path(dataset_root),
        "images": len(images_by_stem),
        "labels": len(labels_by_stem),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "bad_images": bad_images,
        "bad_labels": bad_labels,
        "timeofday_filter": args.timeofday,
        "weather_filter": args.weather,
        "filtered_out_by_scope": filtered_out,
        "num_coco_images": len(coco_images),
        "num_coco_annotations": len(annotations),
        "categories": CATEGORIES,
        "source_timeofday_counts": dict(sorted(timeofday_counts.items(), key=lambda item: str(item[0]))),
        "source_weather_counts": dict(sorted(weather_counts.items(), key=lambda item: str(item[0]))),
        "selected_timeofday_counts": dict(sorted(selected_timeofday_counts.items(), key=lambda item: str(item[0]))),
        "selected_weather_counts": dict(sorted(selected_weather_counts.items(), key=lambda item: str(item[0]))),
        "kept_source_category_counts": dict(sorted(category_counts.items())),
        "ignored_source_category_counts": dict(sorted(ignored_category_counts.items())),
        "invalid_box_counts": dict(sorted(invalid_box_counts.items())),
        "coco_json": norm_path(output_json),
        "manifest_csv": norm_path(manifest_csv),
    }
    (output_dir / "{}_summary.json".format(args.name)).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared BDD100K Faster R-CNN evaluation data")
    print("  timeofday filter: {}".format(args.timeofday))
    print("  weather filter: {}".format(args.weather))
    print("  images: {}".format(len(coco_images)))
    print("  annotations: {}".format(len(annotations)))
    print("  invalid boxes skipped: {}".format(sum(invalid_box_counts.values())))
    print("  json: {}".format(output_json))


if __name__ == "__main__":
    main()

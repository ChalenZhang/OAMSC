"""Convert Foggy Cityscapes into the shared seven-class COCO protocol.

The selected beta setting, retained images, and skipped annotations are recorded.
"""

from __future__ import print_function

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import IMAGE_EXTS, image_size, norm_path


CATEGORIES = [
    {"id": 1, "name": "Bicycle", "source_cityscapes_label": "bicycle"},
    {"id": 2, "name": "Bus", "source_cityscapes_label": "bus"},
    {"id": 3, "name": "Car", "source_cityscapes_label": "car"},
    {"id": 4, "name": "Motor", "source_cityscapes_label": "motorcycle"},
    {"id": 5, "name": "Psn.", "source_cityscapes_label": "person"},
    {"id": 6, "name": "Rider", "source_cityscapes_label": "rider"},
    {"id": 7, "name": "Truck", "source_cityscapes_label": "truck"},
]

CITYSCAPES_TO_CATEGORY_ID = {
    "bicycle": 1,
    "bus": 2,
    "car": 3,
    "motorcycle": 4,
    "person": 5,
    "rider": 6,
    "truck": 7,
}

FOGGY_RE = re.compile(r"^(?P<base>.+)_leftImg8bit_foggy_beta_(?P<beta>[^_]+)$")


def selected_value(value, requested):
    requested = str(requested).strip()
    if requested.lower() in ("", "all", "*", "any"):
        return True
    return str(value) == requested


def collect_images(image_root):
    rows = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = FOGGY_RE.match(path.stem)
        if match is None:
            continue
        info = match.groupdict()
        rows.append(
            {
                "path": path,
                "city": path.parent.name,
                "base": info["base"],
                "beta": info["beta"],
            }
        )
    return rows


def polygon_bbox(poly, width, height):
    if not poly:
        return None
    xs = []
    ys = []
    for point in poly:
        if len(point) < 2:
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))
    if not xs or not ys:
        return None

    x1 = max(0.0, min(float(width), min(xs)))
    y1 = max(0.0, min(float(height), min(ys)))
    x2 = max(0.0, min(float(width), max(xs)))
    y2 = max(0.0, min(float(height), max(ys)))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1.0 or box_h <= 1.0:
        return None
    return [x1, y1, box_w, box_h]


def label_path_for(labels_root, city, base):
    return labels_root / city / "{}_gtFine_polygons.json".format(base)


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "Cityscapes_foggy_val")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_cityscapes_foggy")
    parser.add_argument("--name", default="cityscapes_foggy_val_beta_0.02")
    parser.add_argument("--beta", default="0.02", help="Foggy beta split, or all")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_image_rows = collect_images(image_root)
    selected_rows = [row for row in all_image_rows if selected_value(row["beta"], args.beta)]

    coco_images = []
    annotations = []
    manifest_rows = []
    label_counts = Counter()
    ignored_label_counts = Counter()
    invalid_box_counts = Counter()
    beta_counts = Counter(row["beta"] for row in all_image_rows)
    selected_beta_counts = Counter(row["beta"] for row in selected_rows)
    selected_city_counts = Counter(row["city"] for row in selected_rows)
    missing_labels = []
    bad_images = []
    bad_labels = []
    ann_id = 1
    image_id = 1

    for row in selected_rows:
        image_path = row["path"]
        label_path = label_path_for(labels_root, row["city"], row["base"])
        if not label_path.exists():
            missing_labels.append(
                {
                    "image_path": norm_path(image_path),
                    "expected_label": norm_path(label_path),
                }
            )
            continue

        try:
            width, height = image_size(image_path)
        except Exception as exc:
            bad_images.append(
                {
                    "image_path": norm_path(image_path),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            continue

        try:
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception as exc:
            bad_labels.append(
                {
                    "label_path": norm_path(label_path),
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
                "scene": row["base"],
                "style": "Cityscapes_foggy_val",
                "city": row["city"],
                "beta": row["beta"],
                "label_file": norm_path(label_path),
            }
        )

        kept = 0
        ignored = 0
        invalid = 0
        for obj in label_data.get("objects", []):
            source_label = obj.get("label")
            category_id = CITYSCAPES_TO_CATEGORY_ID.get(source_label)
            if category_id is None:
                ignored_label_counts[source_label] += 1
                ignored += 1
                continue

            bbox = polygon_bbox(obj.get("polygon", []), width, height)
            if bbox is None:
                invalid_box_counts[source_label] += 1
                invalid += 1
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "source_cityscapes_label": source_label,
                }
            )
            ann_id += 1
            kept += 1
            label_counts[source_label] += 1

        manifest_rows.append(
            {
                "image_path": norm_path(image_path),
                "label_path": norm_path(label_path),
                "city": row["city"],
                "scene": row["base"],
                "beta": row["beta"],
                "width": width,
                "height": height,
                "kept_annotations": kept,
                "ignored_annotations": ignored,
                "invalid_annotations": invalid,
            }
        )
        image_id += 1

    output_json = output_dir / "{}_coco.json".format(args.name)
    data = {
        "info": {
            "description": "Cityscapes_foggy_val converted from gtFine polygons to COCO-style boxes",
            "label_note": "Only bicycle, bus, car, motorcycle, person, rider, truck are kept and mapped to the 7 training classes.",
            "beta_filter": args.beta,
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
                "image_path",
                "label_path",
                "city",
                "scene",
                "beta",
                "width",
                "height",
                "kept_annotations",
                "ignored_annotations",
                "invalid_annotations",
            ],
        )
        writer.writeheader()
        for manifest_row in manifest_rows:
            writer.writerow(manifest_row)

    summary = {
        "dataset_root": norm_path(dataset_root),
        "beta_filter": args.beta,
        "source_images": len(all_image_rows),
        "selected_images_before_validation": len(selected_rows),
        "num_coco_images": len(coco_images),
        "num_coco_annotations": len(annotations),
        "missing_labels": missing_labels,
        "bad_images": bad_images,
        "bad_labels": bad_labels,
        "categories": CATEGORIES,
        "source_beta_counts": dict(sorted(beta_counts.items())),
        "selected_beta_counts": dict(sorted(selected_beta_counts.items())),
        "selected_city_counts": dict(sorted(selected_city_counts.items())),
        "kept_source_label_counts": dict(sorted(label_counts.items())),
        "ignored_source_label_counts": dict(sorted(ignored_label_counts.items(), key=lambda item: str(item[0]))),
        "invalid_box_counts": dict(sorted(invalid_box_counts.items())),
        "coco_json": norm_path(output_json),
        "manifest_csv": norm_path(manifest_csv),
    }
    (output_dir / "{}_summary.json".format(args.name)).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared Cityscapes_foggy_val Faster R-CNN evaluation data")
    print("  beta filter: {}".format(args.beta))
    print("  images: {}".format(len(coco_images)))
    print("  annotations: {}".format(len(annotations)))
    print("  missing labels: {}".format(len(missing_labels)))
    print("  json: {}".format(output_json))


if __name__ == "__main__":
    main()

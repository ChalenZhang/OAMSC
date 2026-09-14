"""Convert clean Cityscapes validation data to seven-class COCO records.

The converter preserves source image dimensions and reports class/count checks.
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import IMAGE_EXTS, image_size, norm_path
from prepare_fasterrcnn_cityscapes_foggy import (
    CATEGORIES,
    CITYSCAPES_TO_CATEGORY_ID,
    polygon_bbox,
)


def collect_images(image_root):
    rows = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not path.stem.endswith("_leftImg8bit"):
            continue
        base = path.stem[: -len("_leftImg8bit")]
        rows.append(
            {
                "path": path,
                "city": path.parent.name,
                "base": base,
            }
        )
    return rows


def label_path_for(labels_root, city, base):
    return labels_root / city / "{}_gtFine_polygons.json".format(base)


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "Cityscapes_val")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_cityscapes_val")
    parser.add_argument("--name", default="cityscapes_val")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    image_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rows = collect_images(image_root)
    coco_images = []
    annotations = []
    manifest_rows = []
    label_counts = Counter()
    ignored_label_counts = Counter()
    invalid_box_counts = Counter()
    city_counts = Counter(row["city"] for row in image_rows)
    missing_labels = []
    bad_images = []
    bad_labels = []
    ann_id = 1
    image_id = 1

    for row in image_rows:
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
                "style": "Cityscapes_val",
                "city": row["city"],
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
            "description": "Cityscapes_val converted from gtFine polygons to COCO-style boxes",
            "label_note": "Only bicycle, bus, car, motorcycle, person, rider, truck are kept and mapped to the 7 training classes.",
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
                "width",
                "height",
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
        "num_source_images": len(image_rows),
        "num_images": len(coco_images),
        "num_annotations": len(annotations),
        "source_city_counts": dict(sorted(city_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "ignored_label_counts": dict(sorted((str(k), v) for k, v in ignored_label_counts.items())),
        "invalid_box_counts": dict(sorted(invalid_box_counts.items())),
        "missing_labels": missing_labels,
        "bad_images": bad_images,
        "bad_labels": bad_labels,
        "coco_json": norm_path(output_json),
        "manifest_csv": norm_path(manifest_csv),
    }
    (output_dir / "{}_summary.json".format(args.name)).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared Cityscapes_val Faster R-CNN evaluation data")
    print("  images: {}".format(len(coco_images)))
    print("  annotations: {}".format(len(annotations)))
    print("  cities: {}".format(dict(sorted(city_counts.items()))))
    print("  missing labels: {}".format(len(missing_labels)))
    print("  bad images: {}".format(len(bad_images)))
    print("  bad labels: {}".format(len(bad_labels)))
    print("  coco json: {}".format(output_json))


if __name__ == "__main__":
    main()

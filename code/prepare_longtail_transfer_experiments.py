"""Stage manifest-based YOLO transfer experiments from canonical style roots.

Creates YAML configurations and CSV manifests using canonical style names.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import (
    collect_images as collect_city_images,
    collect_labels as collect_city_labels,
    image_size,
    norm_path,
    read_classes,
    yolo_to_coco_annotations,
)
from prepare_fasterrcnn_fixed_styles import ordered_styles as ordered_city_styles
from prepare_fasterrcnn_realdrivesim_order import (
    collect_images as collect_rds_images,
    collect_labels as collect_rds_labels,
    ordered_styles as ordered_rds_styles,
    paired_stems,
)


DEFAULT_STYLE16 = [
    "Oil-Painting",
    "Sketching",
    "Line-Art",
    "Crayon-Drawing",
    "Ghibli-Anime",
    "American-Comics",
    "Chibi-Comics",
    "Painterly-Anime",
    "3D-Modeling",
    "AAA-Game-Scene",
    "Post-Apocalyptic",
    "Cyberpunk",
    "Collage",
    "Stained-Glass",
    "Paper-Cutting",
    "Textile-Art",
]

CLASS_NAMES = ["Bicycle", "Bus", "Car", "Motor", "Psn.", "Rider", "Truck"]


def safe_name(value):
    out = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("._-") or "item"


def parse_styles(value):
    if value is None or str(value).strip() == "":
        return list(DEFAULT_STYLE16)
    return [part for part in str(value).replace(",", " ").split() if part]


def collect_city(root):
    styles = ordered_city_styles(root)
    images_by_style = {style: collect_city_images(root / style) for style in styles}
    labels_by_style = {style: collect_city_labels(root / style) for style in styles}
    stems = sorted(images_by_style["Origin"].keys())
    return styles, stems, images_by_style, labels_by_style


def collect_rds(root):
    styles = ordered_rds_styles(root)
    images_by_style = {}
    for style in styles:
        images, _ignored = collect_rds_images(root / style)
        images_by_style[style] = images
    labels_by_style = {style: collect_rds_labels(root / style) for style in styles}
    stems = sorted(paired_stems(images_by_style, labels_by_style, "Origin"))
    return styles, stems, images_by_style, labels_by_style


def selected_rows(source, stems, styles, images_by_style, labels_by_style):
    rows = []
    missing = []
    for stem in stems:
        for style in styles:
            if stem not in images_by_style.get(style, {}) or stem not in labels_by_style.get(style, {}):
                missing.append({"source": source, "scene": stem, "style": style})
                continue
            rows.append({"source": source, "scene": stem, "style": style})
    return rows, missing


def make_yolo_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    os.symlink(str(src), str(dst))


def write_yolo_label_from_source(label_path, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label_path, out_path)


def build_dataset(rows, city_data, rds_data, categories, output_dir):
    city_styles, _city_stems, city_images, city_labels = city_data
    rds_styles, _rds_stems, rds_images, rds_labels = rds_data

    images = []
    annotations = []
    manifest_rows = []
    yolo_manifest_rows = []
    ann_id = 1
    image_id = 1
    source_counts = Counter()
    style_counts = Counter()
    scene_style_counts = Counter()
    size_cache = {}
    yolo_root = output_dir / "yolo"
    yolo_image_dir = yolo_root / "images" / "train"
    yolo_label_dir = yolo_root / "labels" / "train"

    for row in rows:
        source = row["source"]
        style = row["style"]
        stem = row["scene"]
        if source == "city":
            image_path = city_images[style][stem]
            label_path = city_labels[style][stem]
        elif source == "rds":
            image_path = rds_images[style][stem]
            label_path = rds_labels[style][stem]
        else:
            raise RuntimeError("Unknown source {}".format(source))

        size_key = norm_path(image_path)
        if size_key not in size_cache:
            size_cache[size_key] = image_size(image_path)
        width, height = size_cache[size_key]
        scene_key = "{}:{}".format(source, stem)
        style_key = "{}:{}".format(source, style)

        images.append(
            {
                "id": image_id,
                "file_name": norm_path(image_path),
                "width": width,
                "height": height,
                "scene": scene_key,
                "style": style_key,
                "label_file": norm_path(label_path),
                "source_dataset": source,
                "source_scene": stem,
                "source_style": style,
            }
        )
        image_annotations, ann_id = yolo_to_coco_annotations(
            label_path, image_id, width, height, ann_id
        )
        annotations.extend(image_annotations)

        yolo_stem = "{}__{}__{}".format(source, safe_name(style), safe_name(stem))
        yolo_image_path = yolo_image_dir / "{}{}".format(yolo_stem, image_path.suffix.lower())
        yolo_label_path = yolo_label_dir / "{}.txt".format(yolo_stem)
        make_yolo_link(image_path, yolo_image_path)
        write_yolo_label_from_source(label_path, yolo_label_path)

        manifest_row = {
            "image_id": image_id,
            "source": source,
            "scene": scene_key,
            "source_scene": stem,
            "style": style_key,
            "source_style": style,
            "image_path": norm_path(image_path),
            "label_path": norm_path(label_path),
            "width": width,
            "height": height,
            "annotations": len(image_annotations),
            "yolo_image_path": norm_path(yolo_image_path),
            "yolo_label_path": norm_path(yolo_label_path),
        }
        manifest_rows.append(manifest_row)
        yolo_manifest_rows.append(manifest_row)
        source_counts[source] += 1
        style_counts[style_key] += 1
        scene_style_counts[scene_key] += 1
        image_id += 1

    coco = {
        "info": {
            "description": "Cityscapes/RDS long-tail transfer experiment data",
            "label_note": "category_id is source YOLO class id + 1; id 0 is reserved for Faster R-CNN background",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    train_json = output_dir / "train_coco.json"
    train_json.write_text(json.dumps(coco, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_csv = output_dir / "selection_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    scene_manifest = yolo_root / "scene_manifest.csv"
    with scene_manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "yolo_image_path",
            "source",
            "scene",
            "source_scene",
            "style",
            "source_style",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in yolo_manifest_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    dataset_yaml = yolo_root / "dataset.yaml"
    names = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    yaml_lines = [
        "path: {}".format(yolo_root),
        "train: images/train",
        "val: images/train",
        "names:",
    ]
    for idx, name in names.items():
        yaml_lines.append("  {}: {}".format(idx, name))
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    multi_style_scene_count = sum(1 for count in scene_style_counts.values() if count >= 2)
    summary = {
        "num_train_images": len(images),
        "num_train_annotations": len(annotations),
        "source_counts": dict(sorted(source_counts.items())),
        "style_counts": dict(sorted(style_counts.items())),
        "multi_style_scene_count": multi_style_scene_count,
        "train_coco_json": norm_path(train_json),
        "manifest_csv": norm_path(manifest_csv),
        "yolo_dataset_yaml": norm_path(dataset_yaml),
        "yolo_scene_manifest": norm_path(scene_manifest),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def experiment_rows(exp_id, city_stems, city_images, city_labels, rds_stems, rds_images, rds_labels, style16):
    origin = ["Origin"]
    origin_plus = ["Origin"] + style16
    if exp_id == 1:
        return selected_rows("city", city_stems, origin, city_images, city_labels)
    if exp_id == 2:
        return selected_rows("rds", rds_stems, origin, rds_images, rds_labels)
    if exp_id == 3:
        a, ma = selected_rows("city", city_stems, origin, city_images, city_labels)
        b, mb = selected_rows("rds", rds_stems, origin, rds_images, rds_labels)
        return a + b, ma + mb
    if exp_id == 4:
        return selected_rows("city", city_stems, origin_plus, city_images, city_labels)
    if exp_id == 5:
        return selected_rows("rds", rds_stems, origin_plus, rds_images, rds_labels)
    if exp_id == 6:
        a, ma = selected_rows("city", city_stems, origin_plus, city_images, city_labels)
        b, mb = selected_rows("rds", rds_stems, origin_plus, rds_images, rds_labels)
        return a + b, ma + mb
    raise RuntimeError("Unknown experiment {}".format(exp_id))


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-root", type=Path, default=project_root.parent / "Cityscapes-Multi-Style")
    parser.add_argument("--rds-root", type=Path, default=project_root.parent / "RealDriveSim-Multi-Style")
    parser.add_argument("--output-root", type=Path, default=project_root / "data_longtail_transfer")
    parser.add_argument("--styles16", default=None, help="Backward-compatible alias for --styles.")
    parser.add_argument("--styles", default=None)
    parser.add_argument("--experiment", type=int, choices=range(1, 7), required=True)
    args = parser.parse_args()

    city_root = args.city_root.resolve()
    rds_root = args.rds_root.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / "exp{:02d}".format(args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)

    style16 = parse_styles(args.styles if args.styles is not None else args.styles16)
    city_styles, city_stems, city_images, city_labels = collect_city(city_root)
    rds_styles, rds_stems, rds_images, rds_labels = collect_rds(rds_root)
    unknown_city = sorted(set(["Origin"] + style16) - set(city_styles))
    unknown_rds = sorted(set(["Origin"] + style16) - set(rds_styles))
    if unknown_city or unknown_rds:
        raise RuntimeError("Unknown styles. City: {}; RDS: {}".format(unknown_city, unknown_rds))

    categories = read_classes(city_root / "classes.txt")
    rows, missing = experiment_rows(
        args.experiment,
        city_stems,
        city_images,
        city_labels,
        rds_stems,
        rds_images,
        rds_labels,
        style16,
    )
    if not rows:
        raise RuntimeError("No rows for experiment {}".format(args.experiment))
    summary = build_dataset(
        rows,
        (city_styles, city_stems, city_images, city_labels),
        (rds_styles, rds_stems, rds_images, rds_labels),
        categories,
        output_dir,
    )
    summary.update(
        {
            "experiment": args.experiment,
            "city_root": norm_path(city_root),
            "rds_root": norm_path(rds_root),
            "styles16": style16,
            "missing_variant_count": len(missing),
            "missing_variants": missing,
        }
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Prepared long-tail transfer experiment exp{:02d}".format(args.experiment))
    print("  train images: {}".format(summary["num_train_images"]))
    print("  annotations: {}".format(summary["num_train_annotations"]))
    print("  multi-style scenes: {}".format(summary["multi_style_scene_count"]))
    print("  coco: {}".format(summary["train_coco_json"]))
    print("  yolo: {}".format(summary["yolo_dataset_yaml"]))


if __name__ == "__main__":
    main()

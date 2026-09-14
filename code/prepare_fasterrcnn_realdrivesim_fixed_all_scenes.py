"""Prepare all RealDriveSim scenes and canonical style views for Faster R-CNN.

The output COCO file and manifests preserve scene/style correspondence and gaps.
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import norm_path, read_classes, style_switches_from_arg
from prepare_fasterrcnn_realdrivesim_order import (
    collect_images,
    collect_labels,
    enabled_training_styles,
    make_coco_dataset,
    ordered_styles,
    paired_stems,
)


def build_all_scene_rows(origin_stems, train_styles, images_by_style, labels_by_style):
    rows = []
    missing_variants = []
    counts = Counter()

    for stem in origin_stems:
        for style in train_styles:
            image_path = images_by_style[style].get(stem)
            label_path = labels_by_style[style].get(stem)
            if image_path is None or label_path is None:
                missing_variants.append(
                    {
                        "scene": stem,
                        "style": style,
                        "missing_image": image_path is None,
                        "missing_label": label_path is None,
                    }
                )
                continue
            rows.append(
                {
                    "scene": stem,
                    "selected_style": style,
                    "available_style_count": len(train_styles),
                }
            )
            counts[style] += 1

    return rows, missing_variants, dict(sorted(counts.items()))


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "RealDriveSim-Multi-Style")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_realdrivesim_all_scenes")
    parser.add_argument(
        "--enabled-styles",
        default=None,
        help="Comma or space separated style names to enable for training.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_styles = ordered_styles(dataset_root)
    if len(all_styles) != 21:
        raise RuntimeError("Expected 21 styles, found {}: {}".format(len(all_styles), all_styles))

    style_switches = style_switches_from_arg(all_styles, args.enabled_styles)
    train_styles, disabled_styles = enabled_training_styles(all_styles, style_switches)
    categories = read_classes(dataset_root / "classes.txt")

    ignored_non_images = {}
    images_by_style = {}
    for style in all_styles:
        images, ignored = collect_images(dataset_root / style)
        images_by_style[style] = images
        if ignored:
            ignored_non_images[style] = ignored
    labels_by_style = {style: collect_labels(dataset_root / style) for style in all_styles}

    origin_stems = sorted(paired_stems(images_by_style, labels_by_style, "Origin"))
    if not origin_stems:
        raise RuntimeError("ORIGIN has no paired image+label scenes")

    rows, missing_variants, planned_counts = build_all_scene_rows(
        origin_stems, train_styles, images_by_style, labels_by_style
    )
    if not rows:
        raise RuntimeError("No train rows were built")

    train_json = output_dir / "train_coco.json"
    train_data, manifest_rows, skipped_rows, actual_counts = make_coco_dataset(
        rows, images_by_style, labels_by_style, categories, train_json
    )

    manifest_csv = output_dir / "selection_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "selected_style",
                "image_path",
                "label_path",
                "available_style_count",
                "width",
                "height",
                "annotations",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    paired_counts = {}
    missing_labels_by_style = {}
    orphan_labels_by_style = {}
    for style in all_styles:
        image_stems = set(images_by_style[style])
        label_stems = set(labels_by_style[style])
        paired_counts[style] = len(image_stems & label_stems)
        missing_labels_by_style[style] = sorted(image_stems - label_stems)
        orphan_labels_by_style[style] = sorted(label_stems - image_stems)

    summary = {
        "dataset_root": norm_path(dataset_root),
        "sampling_strategy": "realdrivesim_fixed_all_origin_scenes_all_selected_styles",
        "num_origin_paired_scenes": len(origin_stems),
        "num_all_styles": len(all_styles),
        "all_styles": all_styles,
        "num_training_styles": len(train_styles),
        "training_styles": train_styles,
        "disabled_training_styles": disabled_styles,
        "style_switches": style_switches,
        "planned_counts": planned_counts,
        "actual_counts": actual_counts,
        "paired_image_label_counts": paired_counts,
        "ignored_non_image_files": ignored_non_images,
        "missing_labels_by_style": missing_labels_by_style,
        "orphan_labels_by_style": orphan_labels_by_style,
        "missing_variant_count": len(missing_variants),
        "missing_variants": missing_variants,
        "skipped_rows": skipped_rows,
        "num_skipped_rows": len(skipped_rows),
        "num_train_images": len(train_data["images"]),
        "num_train_annotations": len(train_data["annotations"]),
        "train_coco_json": norm_path(train_json),
        "manifest_csv": norm_path(manifest_csv),
        "num_classes_without_background": len(categories),
        "num_classes_with_background": len(categories) + 1,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared RealDriveSim all-scenes Faster R-CNN data")
    print("  enabled training styles: {}".format(", ".join(train_styles)))
    if disabled_styles:
        print("  disabled training styles: {}".format(", ".join(disabled_styles)))
    print("  origin paired scenes: {}".format(len(origin_stems)))
    print("  train images: {}".format(len(train_data["images"])))
    print("  train annotations: {}".format(len(train_data["annotations"])))
    print("  missing style variants: {}".format(len(missing_variants)))
    print("  skipped rows: {}".format(len(skipped_rows)))
    print("  train json: {}".format(train_json))
    print("  training style counts:")
    for style in train_styles:
        print("    {:<8} {}".format(style, actual_counts.get(style, 0)))


if __name__ == "__main__":
    main()

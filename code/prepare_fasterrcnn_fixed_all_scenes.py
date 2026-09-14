"""Prepare every available Cityscapes scene and canonical style view.

Missing views remain explicit in the manifest instead of being synthesized.
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import (
    collect_images,
    collect_labels,
    make_coco_dataset,
    norm_path,
    read_classes,
    style_switches_from_arg,
)
from prepare_fasterrcnn_fixed_styles import ordered_styles, enabled_training_styles


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
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "Cityscapes-Multi-Style")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_fixed_all_scenes")
    parser.add_argument("--expected-scenes", type=int, default=2975)
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
    images_by_style = {style: collect_images(dataset_root / style) for style in all_styles}
    labels_by_style = {style: collect_labels(dataset_root / style) for style in all_styles}

    origin_stems = sorted(images_by_style["Origin"].keys())
    if len(origin_stems) != args.expected_scenes:
        raise RuntimeError(
            "ORIGIN has {} scenes, expected {}".format(len(origin_stems), args.expected_scenes)
        )

    rows, missing_variants, counts = build_all_scene_rows(
        origin_stems, train_styles, images_by_style, labels_by_style
    )
    if not rows:
        raise RuntimeError("No train rows were built")

    train_json = output_dir / "train_coco.json"
    train_data = make_coco_dataset(
        rows, images_by_style, labels_by_style, categories, train_json
    )

    manifest_csv = output_dir / "selection_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "selected_style", "image_path", "label_path", "available_style_count"],
        )
        writer.writeheader()
        for row in rows:
            style = row["selected_style"]
            stem = row["scene"]
            writer.writerow(
                {
                    "scene": stem,
                    "selected_style": style,
                    "image_path": norm_path(images_by_style[style][stem]),
                    "label_path": norm_path(labels_by_style[style][stem]),
                    "available_style_count": row["available_style_count"],
                }
            )

    summary = {
        "dataset_root": norm_path(dataset_root),
        "sampling_strategy": "fixed_all_origin_scenes_all_selected_styles",
        "expected_origin_scenes": args.expected_scenes,
        "num_origin_scenes": len(origin_stems),
        "num_all_styles": len(all_styles),
        "num_training_styles": len(train_styles),
        "training_styles": train_styles,
        "disabled_training_styles": disabled_styles,
        "style_switches": style_switches,
        "num_train_images": len(train_data["images"]),
        "num_train_annotations": len(train_data["annotations"]),
        "actual_counts": counts,
        "missing_variant_count": len(missing_variants),
        "missing_variants": missing_variants,
        "train_coco_json": norm_path(train_json),
        "manifest_csv": norm_path(manifest_csv),
        "num_classes_without_background": len(categories),
        "num_classes_with_background": len(categories) + 1,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared fixed all-scenes Faster R-CNN Cityscapes-Multi-Style data")
    print("  enabled training styles: {}".format(", ".join(train_styles)))
    if disabled_styles:
        print("  disabled training styles: {}".format(", ".join(disabled_styles)))
    print("  origin scenes: {}".format(len(origin_stems)))
    print("  train images: {}".format(len(train_data["images"])))
    print("  train annotations: {}".format(len(train_data["annotations"])))
    print("  missing style variants: {}".format(len(missing_variants)))
    print("  train json: {}".format(train_json))
    print("  training style counts:")
    for style in train_styles:
        print("    {:<8} {}".format(style, counts.get(style, 0)))


if __name__ == "__main__":
    main()

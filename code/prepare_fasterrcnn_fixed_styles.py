"""Prepare a deterministic Cityscapes subset with a fixed style budget.

The seed, selected public style names, and scene-level availability are recorded.
"""

from __future__ import print_function

import argparse
import csv
import json
import random
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import (
    collect_images,
    collect_labels,
    make_coco_dataset,
    norm_path,
    read_classes,
    style_switches_from_arg,
)


# Set 1 to include a style in the fixed per-scene training bundle.
# Set 0 to exclude it. Every selected scene contributes exactly one image from
# every enabled style, so style counts are identical by construction.
STYLE_SWITCHES = {
    "Origin": 1,

    "Sketching": 1,
    "Oil-Painting": 1,
    "Chinese-Painting": 1,
    "Line-Art": 1,
    "Crayon-Drawing": 1,

    "Ghibli-Anime": 1,
    "Chibi-Comics": 1,
    "Painterly-Anime": 1,
    "American-Comics": 1,
    "Pixel-Art": 1,

    "AAA-Game-Scene": 1,
    "3D-Modeling": 1,
    "Science-Fiction": 1,
    "Cyberpunk": 1,
    "Post-Apocalyptic": 1,

    "Textile-Art": 1,
    "Paper-Cutting": 1,
    "Stained-Glass": 1,
    "Building-Blocks": 1,
    "Collage": 1,
}


def ordered_styles(dataset_root):
    styles = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if "Origin" in styles:
        styles.remove("Origin")
        styles = ["Origin"] + styles
    return styles


def enabled_training_styles(all_styles, style_switches=None):
    if style_switches is None:
        style_switches = STYLE_SWITCHES

    unknown = sorted(set(style_switches) - set(all_styles))
    if unknown:
        raise RuntimeError("STYLE_SWITCHES contains unknown styles: {}".format(unknown))

    enabled = [style for style in all_styles if int(style_switches.get(style, 1)) == 1]
    disabled = [style for style in all_styles if int(style_switches.get(style, 1)) == 0]
    if not enabled:
        raise RuntimeError("At least one style must be enabled in STYLE_SWITCHES")
    return enabled, disabled


def build_fixed_style_rows(origin_stems, train_styles, images_by_style, seed, expected_scenes):
    eligible_scenes = [
        stem
        for stem in origin_stems
        if all(stem in images_by_style[style] for style in train_styles)
    ]

    n_styles = len(train_styles)
    scenes_to_select = expected_scenes if n_styles == 1 else expected_scenes // n_styles + 1
    total_images = scenes_to_select * n_styles

    if scenes_to_select > len(eligible_scenes):
        raise RuntimeError(
            "Need {} eligible scenes for {} enabled styles, but only {} scenes have all enabled styles.".format(
                scenes_to_select, n_styles, len(eligible_scenes)
            )
        )

    rng = random.Random(seed)
    shuffled = list(eligible_scenes)
    rng.shuffle(shuffled)
    selected_scenes = sorted(shuffled[:scenes_to_select])

    rows = []
    for stem in selected_scenes:
        for style in train_styles:
            rows.append(
                {
                    "scene": stem,
                    "selected_style": style,
                    "available_style_count": n_styles,
                }
            )

    counts = {style: scenes_to_select for style in train_styles}
    return rows, selected_scenes, eligible_scenes, counts, total_images


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "Cityscapes-Multi-Style")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_fixed_styles")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--expected-scenes", type=int, default=2975)
    parser.add_argument(
        "--enabled-styles",
        default=None,
        help="Comma or space separated style names to enable for training. Overrides STYLE_SWITCHES.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "domain_eval").mkdir(parents=True, exist_ok=True)

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

    rows, selected_scenes, eligible_scenes, counts, total_images = build_fixed_style_rows(
        origin_stems, train_styles, images_by_style, args.seed, args.expected_scenes
    )

    print("Reading image sizes from each selected style image...")
    size_cache = {}

    train_json = output_dir / "train_coco.json"
    train_data = make_coco_dataset(
        rows, images_by_style, labels_by_style, categories, train_json, size_cache
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

    eval_jsons = {}
    missing_by_style = {}
    for style in all_styles:
        present_stems = sorted(stem for stem in origin_stems if stem in images_by_style[style])
        missing_by_style[style] = sorted(set(origin_stems) - set(present_stems))
        eval_rows = [
            {"scene": stem, "selected_style": style, "available_style_count": 1}
            for stem in present_stems
        ]
        style_json = output_dir / "domain_eval" / "{}_coco.json".format(style)
        make_coco_dataset(eval_rows, images_by_style, labels_by_style, categories, style_json, size_cache)
        eval_jsons[style] = norm_path(style_json)

    summary = {
        "dataset_root": norm_path(dataset_root),
        "seed": args.seed,
        "sampling_strategy": "fixed_styles_per_random_scene",
        "expected_scenes": args.expected_scenes,
        "num_all_styles": len(all_styles),
        "num_training_styles": len(train_styles),
        "training_styles": train_styles,
        "disabled_training_styles": disabled_styles,
        "style_switches": style_switches,
        "eligible_scene_count": len(eligible_scenes),
        "selected_scene_count": len(selected_scenes),
        "target_minimum_exceeded_images": total_images,
        "num_train_images": len(train_data["images"]),
        "num_train_annotations": len(train_data["annotations"]),
        "actual_counts": counts,
        "missing_by_style": missing_by_style,
        "train_coco_json": norm_path(train_json),
        "manifest_csv": norm_path(manifest_csv),
        "domain_eval_jsons": eval_jsons,
        "num_classes_without_background": len(categories),
        "num_classes_with_background": len(categories) + 1,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Prepared fixed-style Faster R-CNN Cityscapes-Multi-Style data")
    print("  enabled training styles: {}".format(", ".join(train_styles)))
    if disabled_styles:
        print("  disabled training styles: {}".format(", ".join(disabled_styles)))
    print("  selected scenes: {}".format(len(selected_scenes)))
    print("  train images: {}".format(len(train_data["images"])))
    print("  train annotations: {}".format(len(train_data["annotations"])))
    print("  train json: {}".format(train_json))
    print("  training style counts:")
    for style in train_styles:
        print("    {:<8} {}".format(style, counts[style]))


if __name__ == "__main__":
    main()

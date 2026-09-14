"""Prepare ordered or fixed-budget RealDriveSim multi-style experiments.

Public style names, deterministic selection, and integrity statistics are emitted.
"""

from __future__ import print_function

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from prepare_fasterrcnn_cityscapes20 import (
    IMAGE_EXTS,
    image_size,
    norm_path,
    read_classes,
    style_switches_from_arg,
    yolo_to_coco_annotations,
)


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
        raise RuntimeError("At least one style must be enabled")
    return enabled, disabled


def collect_images(style_dir):
    image_dir = style_dir / "images"
    if not image_dir.is_dir():
        raise RuntimeError("Missing images directory: {}".format(image_dir))
    images = {}
    ignored = []
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            ignored.append(path.name)
            continue
        images[path.stem] = path
    return images, ignored


def collect_labels(style_dir):
    label_dir = style_dir / "labels"
    if not label_dir.is_dir():
        raise RuntimeError("Missing labels directory: {}".format(label_dir))
    return {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def paired_stems(images_by_style, labels_by_style, style):
    return set(images_by_style[style]) & set(labels_by_style[style])


def balanced_random_assignments(styles, scene_stems, images_by_style, labels_by_style, seed):
    rng = random.Random(seed)
    base = len(scene_stems) // len(styles)
    remainder = len(scene_stems) % len(styles)

    shuffled_styles = list(styles)
    rng.shuffle(shuffled_styles)
    targets = {style: base for style in styles}
    for style in shuffled_styles[:remainder]:
        targets[style] += 1

    slots = []
    for style in styles:
        slots.extend([style] * targets[style])
    rng.shuffle(slots)

    shuffled_scenes = list(scene_stems)
    rng.shuffle(shuffled_scenes)
    shuffled_scenes.sort(
        key=lambda stem: sum(
            stem in images_by_style[style] and stem in labels_by_style[style]
            for style in styles
        )
    )

    rows = []
    counts = {style: 0 for style in styles}
    for stem in shuffled_scenes:
        candidate_indices = [
            idx
            for idx, style in enumerate(slots)
            if stem in images_by_style[style] and stem in labels_by_style[style]
        ]
        if not candidate_indices:
            raise RuntimeError("No available image+label found for scene {}".format(stem))
        idx = rng.choice(candidate_indices)
        selected_style = slots.pop(idx)
        counts[selected_style] += 1
        rows.append(
            {
                "scene": stem,
                "selected_style": selected_style,
                "available_style_count": sum(
                    stem in images_by_style[style] and stem in labels_by_style[style]
                    for style in styles
                ),
            }
        )
    return rows, targets, counts


def fixed_style_rows(origin_stems, train_styles, images_by_style, labels_by_style, seed, target_images):
    eligible_scenes = [
        stem
        for stem in origin_stems
        if all(stem in images_by_style[style] and stem in labels_by_style[style] for style in train_styles)
    ]
    n_styles = len(train_styles)
    scenes_to_select = target_images if n_styles == 1 else target_images // n_styles + 1
    if scenes_to_select > len(eligible_scenes):
        raise RuntimeError(
            "Need {} eligible scenes for {} enabled styles, but only {} scenes have complete image+label pairs.".format(
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
    return rows, selected_scenes, eligible_scenes, counts


def make_coco_dataset(rows, images_by_style, labels_by_style, categories, output_json):
    images = []
    annotations = []
    manifest_rows = []
    skipped_rows = []
    style_counts = Counter()
    ann_id = 1
    image_id = 1

    for row in rows:
        style = row["selected_style"]
        stem = row["scene"]
        image_path = images_by_style[style].get(stem)
        label_path = labels_by_style[style].get(stem)
        if image_path is None or label_path is None:
            skipped_rows.append(
                {
                    "scene": stem,
                    "style": style,
                    "reason": "missing_image_or_label",
                }
            )
            continue

        try:
            width, height = image_size(image_path)
            image_annotations, next_ann_id = yolo_to_coco_annotations(
                label_path, image_id, width, height, ann_id
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "scene": stem,
                    "style": style,
                    "image_path": norm_path(image_path),
                    "label_path": norm_path(label_path),
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            continue

        images.append(
            {
                "id": image_id,
                "file_name": norm_path(image_path),
                "width": width,
                "height": height,
                "scene": stem,
                "style": style,
                "label_file": norm_path(label_path),
            }
        )
        annotations.extend(image_annotations)
        manifest_rows.append(
            {
                "scene": stem,
                "selected_style": style,
                "image_path": norm_path(image_path),
                "label_path": norm_path(label_path),
                "available_style_count": row["available_style_count"],
                "width": width,
                "height": height,
                "annotations": len(image_annotations),
            }
        )
        style_counts[style] += 1
        ann_id = next_ann_id
        image_id += 1

    data = {
        "info": {
            "description": "RealDriveSim-Multi-Style converted from YOLO txt to COCO-style boxes",
            "label_note": "category_id is source YOLO class id + 1; id 0 is reserved for Faster R-CNN background",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data, manifest_rows, skipped_rows, dict(sorted(style_counts.items()))


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
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
        for row in rows:
            writer.writerow(row)


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "RealDriveSim-Multi-Style")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data_realdrivesim_order")
    parser.add_argument("--algorithm", choices=["fixed", "unfixed"], required=True)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--target-images", type=int, default=6000)
    parser.add_argument(
        "--enabled-styles",
        default=None,
        help="Comma or space separated style names to enable for training. Overrides STYLE_SWITCHES.",
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

    if args.algorithm == "fixed":
        rows, selected_scenes, eligible_scenes, planned_counts = fixed_style_rows(
            origin_stems,
            train_styles,
            images_by_style,
            labels_by_style,
            args.seed,
            args.target_images,
        )
        sampling_strategy = "realdrivesim_fixed_styles_per_random_scene"
        target_counts = planned_counts
        eligible_scene_count = len(eligible_scenes)
        selected_scene_count = len(selected_scenes)
    else:
        rows, target_counts, planned_counts = balanced_random_assignments(
            train_styles,
            origin_stems,
            images_by_style,
            labels_by_style,
            args.seed,
        )
        sampling_strategy = "realdrivesim_random_single_style_per_scene"
        eligible_scene_count = len(origin_stems)
        selected_scene_count = len(origin_stems)

    train_json = output_dir / "train_coco.json"
    train_data, manifest_rows, skipped_rows, actual_counts = make_coco_dataset(
        rows, images_by_style, labels_by_style, categories, train_json
    )

    manifest_csv = output_dir / "selection_manifest.csv"
    write_manifest(manifest_csv, manifest_rows)

    missing_by_style = {}
    orphan_labels_by_style = {}
    paired_counts = {}
    for style in all_styles:
        image_stems = set(images_by_style[style])
        label_stems = set(labels_by_style[style])
        paired_counts[style] = len(image_stems & label_stems)
        missing_by_style[style] = sorted(image_stems - label_stems)
        orphan_labels_by_style[style] = sorted(label_stems - image_stems)

    summary = {
        "dataset_root": norm_path(dataset_root),
        "seed": args.seed,
        "algorithm": args.algorithm,
        "sampling_strategy": sampling_strategy,
        "target_images": args.target_images,
        "num_all_styles": len(all_styles),
        "all_styles": all_styles,
        "num_training_styles": len(train_styles),
        "training_styles": train_styles,
        "disabled_training_styles": disabled_styles,
        "style_switches": style_switches,
        "eligible_scene_count": eligible_scene_count,
        "selected_scene_count": selected_scene_count,
        "planned_counts": planned_counts,
        "target_counts": target_counts,
        "actual_counts": actual_counts,
        "paired_image_label_counts": paired_counts,
        "ignored_non_image_files": ignored_non_images,
        "missing_labels_by_style": missing_by_style,
        "orphan_labels_by_style": orphan_labels_by_style,
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

    print("Prepared RealDriveSim-Multi-Style Faster R-CNN data")
    print("  algorithm: {}".format(args.algorithm))
    print("  enabled training styles: {}".format(", ".join(train_styles)))
    if disabled_styles:
        print("  disabled training styles: {}".format(", ".join(disabled_styles)))
    print("  train images: {}".format(len(train_data["images"])))
    print("  train annotations: {}".format(len(train_data["annotations"])))
    print("  skipped rows: {}".format(len(skipped_rows)))
    print("  train json: {}".format(train_json))
    print("  training style counts:")
    for style in train_styles:
        print("    {:<8} {}".format(style, actual_counts.get(style, 0)))


if __name__ == "__main__":
    main()

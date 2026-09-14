"""Prepare canonical multi-style Cityscapes records for Faster R-CNN.

Scene correspondence is recovered from shared stems, while public English
style names and conversion summaries are written into the COCO metadata.
"""

from __future__ import print_function

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Set 1 to allow a style to participate in random training selection.
# Set 0 to exclude it from training selection. Domain-eval JSON files are still
# generated for every style found in the dataset.
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


def norm_path(path):
    return str(path.resolve()).replace("\\", "/")


def style_switches_from_arg(all_styles, enabled_styles):
    if enabled_styles is None:
        return dict(STYLE_SWITCHES)

    requested = [
        style.strip()
        for style in enabled_styles.replace(",", " ").split()
        if style.strip()
    ]
    unknown = sorted(set(requested) - set(all_styles))
    if unknown:
        raise RuntimeError("--enabled-styles contains unknown styles: {}".format(unknown))
    if not requested:
        raise RuntimeError("--enabled-styles must include at least one style")

    requested_set = set(requested)
    return {style: int(style in requested_set) for style in all_styles}


def enabled_training_styles(all_styles, style_switches=None):
    if style_switches is None:
        style_switches = STYLE_SWITCHES

    unknown = sorted(set(style_switches) - set(all_styles))
    if unknown:
        raise RuntimeError("STYLE_SWITCHES contains unknown styles: {}".format(unknown))

    disabled = [style for style in all_styles if int(style_switches.get(style, 1)) == 0]
    enabled = [style for style in all_styles if int(style_switches.get(style, 1)) == 1]
    if not enabled:
        raise RuntimeError("At least one style must be enabled in STYLE_SWITCHES")
    return enabled, disabled


def read_classes(classes_path):
    categories = []
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        source_id = int(parts[0])
        name = parts[1] if len(parts) > 1 else str(source_id)
        categories.append({"id": source_id + 1, "name": name, "source_yolo_id": source_id})
    if not categories:
        raise RuntimeError("No classes found in {}".format(classes_path))
    return categories


def collect_images(style_dir):
    image_dir = style_dir / "images"
    if not image_dir.is_dir():
        raise RuntimeError("Missing images directory: {}".format(image_dir))
    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def collect_labels(style_dir):
    label_dir = style_dir / "labels"
    if not label_dir.is_dir():
        raise RuntimeError("Missing labels directory: {}".format(label_dir))
    return {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def image_size(path):
    with Image.open(path) as img:
        return img.size


def yolo_to_coco_annotations(label_path, image_id, width, height, next_ann_id):
    annotations = []
    if not label_path.exists():
        raise RuntimeError("Missing label file: {}".format(label_path))

    for line_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise RuntimeError("Bad label line {} in {}: {}".format(line_idx, label_path, line))

        cls = int(float(parts[0]))
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        box_w = bw * width
        box_h = bh * height
        x1 = (xc * width) - box_w / 2.0
        y1 = (yc * height) - box_h / 2.0
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        box_w = max(0.0, min(float(width) - x1, box_w))
        box_h = max(0.0, min(float(height) - y1, box_h))
        if box_w <= 1.0 or box_h <= 1.0:
            continue

        annotations.append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": cls + 1,
                "bbox": [x1, y1, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
                "source_yolo_class": cls,
            }
        )
        next_ann_id += 1
    return annotations, next_ann_id


def balanced_random_assignments(styles, scene_stems, images_by_style, seed):
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
    shuffled_scenes.sort(key=lambda stem: sum(stem in images_by_style[s] for s in styles))

    rows = []
    counts = {style: 0 for style in styles}
    for stem in shuffled_scenes:
        candidate_indices = [i for i, style in enumerate(slots) if stem in images_by_style[style]]
        if not candidate_indices:
            raise RuntimeError("No available style image found for scene {}".format(stem))
        idx = rng.choice(candidate_indices)
        selected_style = slots.pop(idx)
        counts[selected_style] += 1
        rows.append(
            {
                "scene": stem,
                "selected_style": selected_style,
                "available_style_count": sum(stem in images_by_style[s] for s in styles),
            }
        )
    return rows, targets, counts


def make_coco_dataset(rows, images_by_style, labels_by_style, categories, output_json, size_cache=None):
    images = []
    annotations = []
    ann_id = 1
    if size_cache is None:
        size_cache = {}
    for image_id, row in enumerate(rows, 1):
        style = row["selected_style"]
        stem = row["scene"]
        image_path = images_by_style[style][stem]
        label_path = labels_by_style[style].get(stem)
        if label_path is None:
            raise RuntimeError("Missing label for {} in style {}".format(stem, style))
        size_key = norm_path(image_path)
        if size_key not in size_cache:
            size_cache[size_key] = image_size(image_path)
        width, height = size_cache[size_key]
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
        image_annotations, ann_id = yolo_to_coco_annotations(
            label_path, image_id, width, height, ann_id
        )
        annotations.extend(image_annotations)

    data = {
        "info": {
            "description": "Cityscapes-Multi-Style converted from YOLO txt to COCO-style boxes",
            "label_note": "category_id is source YOLO class id + 1; id 0 is reserved for Faster R-CNN background",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def main():
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=workspace_root / "Cityscapes-Multi-Style")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data")
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

    style_dirs = [path for path in dataset_root.iterdir() if path.is_dir()]
    all_styles = sorted(path.name for path in style_dirs)
    if "Origin" in all_styles:
        all_styles.remove("Origin")
        all_styles = ["Origin"] + all_styles
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

    assignments, targets, counts = balanced_random_assignments(
        train_styles, origin_stems, images_by_style, args.seed
    )

    print("Reading image sizes from each selected style image...")
    size_cache = {}

    train_json = output_dir / "train_coco.json"
    train_data = make_coco_dataset(
        assignments, images_by_style, labels_by_style, categories, train_json, size_cache
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
            ],
        )
        writer.writeheader()
        for row in assignments:
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
        "sampling_strategy": "random_single_style_per_scene",
        "num_train_images": len(train_data["images"]),
        "num_train_annotations": len(train_data["annotations"]),
        "num_all_styles": len(all_styles),
        "num_training_styles": len(train_styles),
        "all_styles": all_styles,
        "training_styles": train_styles,
        "disabled_training_styles": disabled_styles,
        "style_switches": style_switches,
        "target_counts": targets,
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

    print("Prepared Faster R-CNN Cityscapes-Multi-Style data")
    print("  train images: {}".format(len(train_data["images"])))
    print("  train annotations: {}".format(len(train_data["annotations"])))
    print("  train json: {}".format(train_json))
    print("  enabled training styles: {}".format(", ".join(train_styles)))
    if disabled_styles:
        print("  disabled training styles: {}".format(", ".join(disabled_styles)))
    print("  training style counts:")
    for style in train_styles:
        print("    {:<8} {}".format(style, counts[style]))


if __name__ == "__main__":
    main()

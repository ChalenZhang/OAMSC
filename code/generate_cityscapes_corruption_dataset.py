"""Materialize deterministic Cityscapes corruptions for offline evaluation.

The generator writes the requested corruption/severity shards and a summary
that downstream evaluators use to verify image counts.
"""

from __future__ import print_function

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_cityscapes_corruption_benchmark import (
    CORRUPTIONS,
    apply_corruption,
    patch_imagecorruptions_compat,
    stable_seed,
)


def image_rel_path(info):
    source = Path(info["file_name"])
    city = info.get("city") or source.parent.name or "unknown_city"
    return Path(city) / "{}.png".format(source.stem)


def condition_annotation_name(corruption, severity):
    if corruption == "Clean":
        return "Clean_0.json"
    return "{}_severity_{}.json".format(corruption, int(severity))


def corrupted_image_path(root, info, corruption, severity):
    return (
        Path(root)
        / "images"
        / corruption
        / "severity_{}".format(int(severity))
        / image_rel_path(info)
    )


def build_annotation_json(data, root, corruption, severity):
    updated = dict(data)
    images = []
    for info in data["images"]:
        new_info = dict(info)
        if corruption != "Clean":
            new_info["file_name"] = str(corrupted_image_path(root, info, corruption, severity))
            new_info["corruption"] = corruption
            new_info["severity"] = int(severity)
        images.append(new_info)
    updated["images"] = images
    updated["corruption_benchmark"] = {
        "backend": "imagecorruptions",
        "corruption": corruption,
        "severity": int(severity),
        "note": "Corrupted images are generated once and reused for all models.",
    }
    return updated


def write_annotation(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_one(task):
    source_path, out_path, corruption, severity, seed, overwrite = task
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    patch_imagecorruptions_compat()

    image = Image.open(source_path).convert("RGB")
    corrupted = apply_corruption(image, corruption, severity, "imagecorruptions")
    corrupted.save(out_path, format="PNG", compress_level=1)
    return "generated"


def make_tasks(data, root, overwrite):
    tasks = []
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            for info in data["images"]:
                seed = stable_seed(info["file_name"], corruption, severity)
                tasks.append(
                    (
                        info["file_name"],
                        str(corrupted_image_path(root, info, corruption, severity)),
                        corruption,
                        severity,
                        seed,
                        overwrite,
                    )
                )
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--annotation-only", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("COCO json not found: {}".format(args.data))

    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out_root.mkdir(parents=True, exist_ok=True)
    annotation_dir = args.out_root / "annotations"
    write_annotation(annotation_dir / "Clean_0.json", build_annotation_json(data, args.out_root, "Clean", 0))
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            write_annotation(
                annotation_dir / condition_annotation_name(corruption, severity),
                build_annotation_json(data, args.out_root, corruption, severity),
            )

    manifest = {
        "source_data": str(args.data.resolve()),
        "out_root": str(args.out_root.resolve()),
        "backend": "imagecorruptions",
        "seed_rule": "stable_seed(file_name, corruption, severity)",
        "num_clean_images": len(data["images"]),
        "corruptions": CORRUPTIONS,
        "severities": [1, 2, 3, 4, 5],
        "annotation_dir": str(annotation_dir.resolve()),
    }
    (args.out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.annotation_only:
        print("Wrote annotations only under {}".format(annotation_dir))
        return

    tasks = make_tasks(data, args.out_root, args.overwrite)
    total = len(tasks)
    generated = 0
    skipped = 0
    print("Generating {} corrupted images with {} workers".format(total, args.workers))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "generated":
                generated += 1
            else:
                skipped += 1
            if index == 1 or index % 100 == 0 or index == total:
                print(
                    "[{}/{}] generated={} skipped={}".format(
                        index, total, generated, skipped
                    ),
                    flush=True,
                )

    print("Done. Dataset root: {}".format(args.out_root))
    print("Annotations: {}".format(annotation_dir))


if __name__ == "__main__":
    main()

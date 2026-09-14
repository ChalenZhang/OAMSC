"""Evaluate Faster R-CNN on a precomputed Cityscapes Corruption tree.

Inputs are prepared COCO records and local corruption images; outputs record
per-corruption scores together with mPC and rPC aggregates.
"""

from __future__ import print_function

import argparse
import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from evaluate_cityscapes_corruption_benchmark import (
    CORRUPTIONS,
    DISPLAY_NAMES,
    SUMMARY_COLUMNS,
    CorruptedCocoDataset,
    append_raw_row,
    discover_weight_files,
    evaluate_condition,
    load_completed,
    metric_columns,
    run_name,
    write_summary_tables,
)
from train_fasterrcnn_resnet101 import build_model, load_checkpoint


def manifest_weight_files(manifest_path):
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint = Path(row.get("checkpoint", ""))
            if not checkpoint.exists():
                raise SystemExit("Missing checkpoint from manifest: {}".format(checkpoint))
            rows.append((row.get("run") or checkpoint.parent.name, checkpoint))
    return rows


def condition_annotation_name(corruption, severity):
    if corruption == "Clean":
        return "Clean_0.json"
    return "{}_severity_{}.json".format(corruption, int(severity))


def condition_json(precomputed_root, corruption, severity):
    return Path(precomputed_root) / "annotations" / condition_annotation_name(corruption, severity)


def write_manifest(out_dir, args, weight_files):
    dependency_versions = {}
    for package in ["imagecorruptions", "scikit-image", "torch", "torchvision", "torchmetrics"]:
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    payload = {
        "precomputed_root": str(Path(args.precomputed_root).resolve()),
        "project": str(Path(args.project).resolve()),
        "run_glob": args.run_glob,
        "num_models": len(weight_files),
        "models": [run_name(path) for path in weight_files],
        "corruptions": CORRUPTIONS,
        "severities": [1, 2, 3, 4, 5],
        "columns": SUMMARY_COLUMNS,
        "metric_columns": metric_columns(),
        "dependency_versions": dependency_versions,
        "metric_note": "Clean is evaluated once. Each corruption column is the mean over severities 1-5. mPC is the mean over the 15 corruption columns. rPC is 100*mPC/Clean.",
        "data_note": "Corrupted images are precomputed once and reused for every model.",
    }
    (out_dir / "benchmark_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_compat_note(out_dir):
    note = {
        "corruption_columns": DISPLAY_NAMES,
        "fairness_note": (
            "All models read the same precomputed corrupted image files for each "
            "corruption/severity condition, avoiding per-model random regeneration."
        ),
    }
    (out_dir / "precomputed_eval_note.json").write_text(
        json.dumps(note, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=project_root / "runs" / "train" / "cityscapes_corruption_precomputed")
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--detections-per-img", type=int, default=300)
    parser.add_argument("--metric-max-detections", type=int, default=100)
    parser.add_argument("--trainable-layers", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--weights-type", choices=["auto", "model", "ema"], default="auto")
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    clean_json = condition_json(args.precomputed_root, "Clean", 0)
    if not clean_json.exists():
        raise SystemExit("Precomputed clean annotation not found: {}".format(clean_json))

    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise SystemExit("torchmetrics detection mAP backend unavailable: {}".format(exc))

    if args.manifest is not None:
        weight_items = manifest_weight_files(args.manifest)
        weight_files = [path for _, path in weight_items]
    else:
        weight_files = discover_weight_files(args.project, args.run_glob)
        weight_items = [(run_name(path), path) for path in weight_files]
    if args.limit_runs > 0:
        weight_items = weight_items[: args.limit_runs]
        weight_files = [path for _, path in weight_items]
    if not weight_items:
        raise SystemExit("No last.pth files found under {} with run glob {}".format(args.project, args.run_glob))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "raw_results.csv"
    completed = load_completed(raw_csv)
    device = torch.device(args.device)

    clean_dataset = CorruptedCocoDataset(clean_json, corruption="Clean", severity=0, corruption_backend="imagecorruptions")
    num_classes = len(clean_dataset.categories) + 1

    write_manifest(args.out_dir, args, weight_files)
    append_compat_note(args.out_dir)

    conditions = [("Clean", 0)]
    if not args.clean_only:
        for corruption in CORRUPTIONS:
            for severity in [1, 2, 3, 4, 5]:
                conditions.append((corruption, severity))

    for corruption, severity in conditions:
        json_path = condition_json(args.precomputed_root, corruption, severity)
        if not json_path.exists():
            raise SystemExit("Missing precomputed annotation: {}".format(json_path))

    for weight_idx, (run, weight_path) in enumerate(weight_items, 1):
        print("[{}/{}] Loading {}".format(weight_idx, len(weight_items), run))
        model = build_model(
            num_classes=num_classes,
            trainable_layers=args.trainable_layers,
            min_size=args.min_size,
            max_size=args.max_size,
            pretrained_backbone=False,
        )
        load_checkpoint(weight_path, model, device=device, weights=args.weights_type)
        model.roi_heads.score_thresh = args.score_threshold
        model.roi_heads.detections_per_img = args.detections_per_img
        model.to(device)
        model.eval()

        for condition_idx, (corruption, severity) in enumerate(conditions, 1):
            key = (run, corruption, severity)
            if key in completed:
                print("  [{}/{}] Skip {} severity {}".format(condition_idx, len(conditions), corruption, severity))
                continue
            print("  [{}/{}] Evaluating {} severity {}".format(condition_idx, len(conditions), corruption, severity))
            dataset = CorruptedCocoDataset(
                condition_json(args.precomputed_root, corruption, severity),
                corruption="Clean",
                severity=0,
                corruption_backend="imagecorruptions",
            )
            metrics = evaluate_condition(args, model, dataset, device, MeanAveragePrecision)
            row = {"run": run, "corruption": corruption, "severity": severity}
            row.update(metrics)
            append_raw_row(raw_csv, row)
            completed.add(key)
            write_summary_tables(args.out_dir, raw_csv)
            print("    map50_95={} map50={}".format(row["map50_95"], row["map50"]))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tables = write_summary_tables(args.out_dir, raw_csv)
    print("Saved raw results: {}".format(raw_csv))
    print("Saved summary tables:")
    for path in tables:
        print("  {}".format(path))


if __name__ == "__main__":
    main()

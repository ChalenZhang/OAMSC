"""Evaluate a released EMA detector using a fixed, documented COCO protocol.

The model loads weights locally. Output contains aggregate metrics and model IDs.
"""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from evaluate_bdd100k_all import CocoStyleDetectionDataset, evaluate_one

ROOT = Path(__file__).resolve().parents[1]
CLASS_ALIASES = (
    {"bicycle", "bike"}, {"bus"}, {"car"}, {"motor", "motorcycle"},
    {"psn.", "person"}, {"rider"}, {"truck"},
)


def validate_categories(categories):
    if len(categories) != 7 or {c["id"] for c in categories} != set(range(1, 8)):
        raise ValueError("Expected seven categories with IDs 1 through 7.")
    for category in categories:
        if category["name"].lower() not in CLASS_ALIASES[category["id"] - 1]:
            raise ValueError("Category names/order differ from the released detector.")


def main():
    registry = json.loads((ROOT / "reproducibility/models.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(registry["models"]), required=True)
    parser.add_argument("--benchmark", choices=list(registry["benchmarks"]), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Locally prepared COCO JSON.")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0, help="Smoke test only; 0 uses the full split.")
    args = parser.parse_args()
    if args.limit_images < 0:
        parser.error("--limit-images must be nonnegative")
    model_info = registry["models"][args.model]
    weights = args.weights_dir / model_info["file"]
    if not weights.is_file():
        parser.error("Model file is missing. Extract the evaluation bundle into its root first.")
    # Restricted deserialization also rejects accidental use of training objects.
    payload = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if set(payload) != {"epoch", "model_ema"}:
        parser.error("Use the tensor-only inference checkpoint supplied in the bundle.")
    if payload["model_ema"]["updates"] != model_info["ema_updates"]:
        parser.error("EMA update count does not match the selected model card.")
    del payload
    dataset = CocoStyleDetectionDataset(args.data, train=False)
    validate_categories(dataset.categories)
    expected_count = registry["benchmarks"][args.benchmark]["images"]
    if len(dataset) != expected_count and not args.limit_images:
        parser.error("Image count differs from the reference split: expected {}, received {}.".format(
            expected_count, len(dataset)))
    if args.limit_images:
        categories = dataset.categories
        dataset = Subset(dataset, range(min(len(dataset), args.limit_images)))
        dataset.categories = categories
    protocol = dict(registry["evaluation"])
    protocol["workers"] = args.workers
    torch.set_num_threads(4)
    metrics = evaluate_one(SimpleNamespace(**protocol), weights, dataset,
                           torch.device(args.device), MeanAveragePrecision)
    # evaluate_one uses paths for local experiment tracking; exclude those fields
    # from the portable result record.
    metrics.pop("run", None)
    metrics.pop("weights", None)
    row = {"model": args.model, "benchmark": args.benchmark,
           "smoke_test": bool(args.limit_images), **metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    if args.limit_images:
        print("Smoke test only: these metrics are not comparable to full-split results.")


if __name__ == "__main__":
    main()

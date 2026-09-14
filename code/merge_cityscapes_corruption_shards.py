"""Merge Cityscapes Corruption shard outputs and regenerate final summaries.

The merger checks shard metadata before combining per-condition measurements.
"""

from __future__ import print_function

import argparse
import csv
import json
import shutil
from pathlib import Path

from evaluate_cityscapes_corruption_benchmark import (
    CORRUPTIONS,
    write_summary_tables,
)


def condition_order():
    order = {("Clean", 0): 0}
    idx = 1
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            order[(corruption, severity)] = idx
            idx += 1
    return order


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = args.out_dir / "raw_results.csv"

    rows = []
    fieldnames = None
    seen = set()
    for shard_dir in args.shard:
        raw_path = shard_dir / "raw_results.csv"
        if not raw_path.exists():
            raise SystemExit("Missing shard raw results: {}".format(raw_path))
        shard_rows = read_rows(raw_path)
        if not shard_rows:
            continue
        if fieldnames is None:
            with raw_path.open(newline="", encoding="utf-8") as f:
                fieldnames = csv.DictReader(f).fieldnames
        for row in shard_rows:
            key = (row["run"], row["corruption"], int(row["severity"]))
            if key in seen:
                raise SystemExit("Duplicate result row while merging: {}".format(key))
            seen.add(key)
            rows.append(row)

    if fieldnames is None:
        raise SystemExit("No rows found in shards")

    cond_order = condition_order()
    rows.sort(
        key=lambda row: (
            row["run"],
            cond_order.get((row["corruption"], int(row["severity"])), 9999),
        )
    )

    with raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary_tables(args.out_dir, raw_out)

    manifest_paths = [shard / "benchmark_manifest.json" for shard in args.shard]
    manifests = []
    for path in manifest_paths:
        if path.exists():
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
    (args.out_dir / "shard_manifests.json").write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for shard_dir in args.shard:
        log_path = shard_dir / "benchmark.log"
        if log_path.exists():
            shutil.copy2(log_path, args.out_dir / "{}.log".format(shard_dir.name))

    print("Merged {} rows into {}".format(len(rows), raw_out))
    print("Saved summary tables under {}".format(args.out_dir / "tables"))


if __name__ == "__main__":
    main()

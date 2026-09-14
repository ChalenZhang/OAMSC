#!/usr/bin/env python3
"""Merge non-overlapping checkpoint-evaluation CSV shards.

Duplicate identities are rejected so a merged table cannot silently double-count.
"""

from __future__ import print_function

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    order = {
        (row["run"], row["checkpoint"]): idx
        for idx, row in enumerate(manifest)
    }

    fieldnames = None
    rows = []
    seen = set()
    for shard in args.shard:
        if not shard.exists():
            raise SystemExit("Missing evaluation shard: {}".format(shard))
        shard_fields, shard_rows = read_rows(shard)
        if fieldnames is None:
            fieldnames = shard_fields
        elif shard_fields != fieldnames:
            raise SystemExit("CSV field mismatch in {}".format(shard))
        for row in shard_rows:
            key = (row["run"], row["checkpoint"])
            if key in seen:
                raise SystemExit("Duplicate evaluation row: {}".format(key))
            if key not in order:
                raise SystemExit("Shard row is absent from manifest: {}".format(key))
            seen.add(key)
            rows.append(row)

    missing = [key for key in order if key not in seen]
    if missing:
        raise SystemExit("Missing {} evaluation rows: {}".format(len(missing), missing))
    rows.sort(key=lambda row: order[(row["run"], row["checkpoint"])])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Merged {} rows into {}".format(len(rows), args.out))


if __name__ == "__main__":
    main()

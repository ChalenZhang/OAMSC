#!/usr/bin/env python3
"""Summarize origin-anchor ablation CSV files into comparison metrics.

Aggregation and percentage conversion are kept separate from raw evaluation.
"""

from __future__ import print_function

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_by_run(path):
    return {row["run"]: row for row in read_csv(path)}


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def fmt(value):
    return "{:.4f}".format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bdd", type=Path, required=True)
    parser.add_argument("--foggy", type=Path, required=True)
    parser.add_argument("--rainy", type=Path, required=True)
    parser.add_argument("--cityc-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    bdd = rows_by_run(args.bdd)
    foggy = rows_by_run(args.foggy)
    rainy = rows_by_run(args.rainy)
    cityc_5095 = rows_by_run(args.cityc_dir / "tables" / "map50_95.csv")
    cityc_50 = rows_by_run(args.cityc_dir / "tables" / "map50.csv")

    rows = []
    for item in manifest:
        run = item["run"]
        missing = [
            name
            for name, table in [
                ("BDD", bdd),
                ("Foggy", foggy),
                ("Rainy", rainy),
                ("Cityscapes-C AP50:95", cityc_5095),
                ("Cityscapes-C AP50", cityc_50),
            ]
            if run not in table
        ]
        if missing:
            raise SystemExit("{} missing results: {}".format(run, ", ".join(missing)))

        bdd_5095 = number(bdd[run], "map50_95")
        fog_5095 = number(foggy[run], "map50_95")
        rain_5095 = number(rainy[run], "map50_95")
        bdd_50 = number(bdd[run], "map50")
        fog_50 = number(foggy[run], "map50")
        rain_50 = number(rainy[run], "map50")
        c95 = cityc_5095[run]
        c50 = cityc_50[run]
        rows.append(
            {
                "exp": item["exp"],
                "description": item.get("description", ""),
                "run": run,
                "bdd_map50_95": bdd_5095,
                "bdd_map50": bdd_50,
                "foggy_map50_95": fog_5095,
                "foggy_map50": fog_50,
                "rainy_map50_95": rain_5095,
                "rainy_map50": rain_50,
                "avg3_map50_95": (bdd_5095 + fog_5095 + rain_5095) / 3.0,
                "avg3_map50": (bdd_50 + fog_50 + rain_50) / 3.0,
                "cityc_clean_map50_95": number(c95, "Clean"),
                "cityc_mpc_map50_95": number(c95, "mPC"),
                "cityc_rpc_map50_95": number(c95, "rPC"),
                "cityc_clean_map50": number(c50, "Clean"),
                "cityc_mpc_map50": number(c50, "mPC"),
                "cityc_rpc_map50": number(c50, "rPC"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "dg_anchor_ablation_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out_dir / "dg_anchor_ablation_summary.md"
    lines = [
        "# City17 ORIGIN-anchored consistency ablation",
        "",
        "All newly trained runs use the same 17-style all-scenes dataset, global batch "
        "size 16, two GPUs, and approximately 20K optimizer steps. A0 reuses the "
        "existing checkpoint.",
        "",
        "| Exp | Design | BDD mAP50:95 | BDD mAP50 | Fog mAP50:95 | Fog mAP50 | "
        "Rain mAP50:95 | Rain mAP50 | Avg3 mAP50:95 | Avg3 mAP50 | "
        "City-C mPC50:95 | City-C rPC50:95 | City-C mPC50 | City-C rPC50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {exp} | {description} | {bdd_map50_95} | {bdd_map50} | "
            "{foggy_map50_95} | {foggy_map50} | {rainy_map50_95} | "
            "{rainy_map50} | {avg3_map50_95} | {avg3_map50} | "
            "{cityc_mpc_map50_95} | {cityc_rpc_map50_95} | "
            "{cityc_mpc_map50} | {cityc_rpc_map50} |".format(
                **{
                    key: fmt(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
        )
    lines.extend(
        [
            "",
            "mPC is averaged over 15 corruptions and five severity levels. "
            "rPC is `100 * mPC / Clean`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved {}".format(csv_path))
    print("Saved {}".format(md_path))


if __name__ == "__main__":
    main()

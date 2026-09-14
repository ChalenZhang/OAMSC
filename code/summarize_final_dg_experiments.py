#!/usr/bin/env python3
"""Assemble final domain-generalization result CSVs into comparison tables.

The script validates expected rows before applying display rounding.
"""

from __future__ import print_function

import argparse
import csv
import math
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
    if not math.isfinite(value):
        return ""
    return "{:.4f}".format(value)


def class_ap50_columns(row):
    return [key for key in row if key.startswith("AP50_")]


def append_class_table(lines, title, manifest, table, columns):
    lines.extend(
        [
            "",
            "## {}".format(title),
            "",
            "| Exp | " + " | ".join(key.replace("AP50_", "") for key in columns) + " |",
            "|---|" + "|".join("---:" for _ in columns) + "|",
        ]
    )
    for item in manifest:
        row = table[item["run"]]
        lines.append(
            "| {} | {} |".format(
                item["exp"],
                " | ".join(fmt(number(row, key)) for key in columns),
            )
        )


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
    tables = [
        ("BDD", bdd),
        ("Foggy", foggy),
        ("Rainy", rainy),
        ("Cityscapes-C AP50:95", cityc_5095),
        ("Cityscapes-C AP50", cityc_50),
    ]

    for item in manifest:
        missing = [name for name, table in tables if item["run"] not in table]
        if missing:
            raise SystemExit(
                "{} missing results: {}".format(item["run"], ", ".join(missing))
            )

    rows = []
    for item in manifest:
        run = item["run"]
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
                "checkpoint_tag": item.get("checkpoint_tag", ""),
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

    baseline = rows[0]
    for row in rows:
        row["delta_avg3_map50_95_vs_f0"] = (
            row["avg3_map50_95"] - baseline["avg3_map50_95"]
        )
        row["delta_avg3_map50_vs_f0"] = (
            row["avg3_map50"] - baseline["avg3_map50"]
        )
        row["delta_cityc_mpc_map50_95_vs_f0"] = (
            row["cityc_mpc_map50_95"] - baseline["cityc_mpc_map50_95"]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "final_dg_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.out_dir / "final_dg_summary.md"
    lines = [
        "# Final City17 domain-generalization experiments",
        "",
        "All primary results use the fixed training endpoint (`last.pth`). "
        "Target-domain test results are not used for checkpoint selection.",
        "",
        "| Exp | Design | BDD 50:95 | BDD 50 | Fog 50:95 | Fog 50 | "
        "Rain 50:95 | Rain 50 | Avg3 50:95 | Avg3 50 | "
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
            "## Improvement over F0",
            "",
            "| Exp | Avg3 mAP50:95 delta | Avg3 mAP50 delta | "
            "City-C mPC50:95 delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} |".format(
                row["exp"],
                fmt(row["delta_avg3_map50_95_vs_f0"]),
                fmt(row["delta_avg3_map50_vs_f0"]),
                fmt(row["delta_cityc_mpc_map50_95_vs_f0"]),
            )
        )

    ap50_columns = class_ap50_columns(next(iter(bdd.values())))
    append_class_table(lines, "BDD daytime clear AP50 by class", manifest, bdd, ap50_columns)
    append_class_table(lines, "Foggy Cityscapes AP50 by class", manifest, foggy, ap50_columns)
    append_class_table(lines, "Rainy Cityscapes AP50 by class", manifest, rainy, ap50_columns)
    lines.extend(
        [
            "",
            "Cityscapes-C mPC is averaged over 15 corruptions and five severity "
            "levels. rPC is `100 * mPC / Clean`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved {}".format(csv_path))
    print("Saved {}".format(md_path))


if __name__ == "__main__":
    main()

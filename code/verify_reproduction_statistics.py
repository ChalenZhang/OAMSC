"""Recompute the released aggregate summaries using the Python standard library."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

DISPLAY = dict(zip(
    ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
     "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
     "elastic_transform", "jpeg_compression", "pixelate"],
    ["Gauss.", "Shot", "Impulse", "Defocus", "Glass", "Motion", "Zoom", "Snow",
     "Frost", "Fog", "Bright", "Contrast", "Elastic", "JPEG", "Pixel"],
))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check_equal(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Aggregate mismatch: {}".format(label))


def verify(root):
    raw = read_csv(root / "corruption_by_severity.csv")
    keys = [(row["corruption"], int(row["severity"])) for row in raw]
    expected_keys = {("Clean", 0)} | {(name, severity) for name in DISPLAY for severity in range(1, 6)}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Expected Clean plus exactly 15 corruptions at five severities.")
    fields = {("seven-class", "AP50"): "map50", ("seven-class", "AP50:95"): "map50_95",
              ("car", "AP50:95"): "AP50_95_Car"}
    for summary in read_csv(root / "corruption_summary.csv"):
        field = fields[(summary["class_scope"], summary["metric"])]
        clean = float(next(row[field] for row in raw if row["corruption"] == "Clean"))
        values = []
        for name, display in DISPLAY.items():
            value = mean(float(row[field]) for row in raw if row["corruption"] == name)
            check_equal(value, summary[display], display)
            values.append(value)
        mpc = mean(values)
        rpc = 100 * mpc / clean
        for key, value in [("Clean", clean), ("mPC", mpc), ("rPC", rpc)]:
            check_equal(value, summary[key], key)
        print("{} {}: Clean={:.2f}, mPC={:.2f}, rPC={:.2f}".format(
            summary["class_scope"], summary["metric"], 100 * clean, 100 * mpc, rpc))
    for row in read_csv(root / "reference_metrics.csv"):
        aps = [float(value) for key, value in row.items() if key.startswith("AP50_")]
        if len(aps) != 7:
            raise ValueError("Expected seven classwise AP50 values.")
        check_equal(mean(aps), row["map50"], row["benchmark"])
    counts = read_csv(root / "style_counts.csv")
    for source, origin in [("Cityscapes", 2975), ("RealDriveSim", 6000)]:
        selected = [row for row in counts if row["source"] == source]
        if len(selected) != 21 or len({row["style"] for row in selected}) != 21:
            raise ValueError("Expected Origin plus 20 distinct styles.")
        for row in selected:
            if int(row["retained_images"]) + int(row["missing_views"]) != origin:
                raise ValueError("Inconsistent retained/missing count.")
    print("Aggregate records are internally consistent. This does not rerun detector inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reproducibility/statistics")
    verify(parser.parse_args().statistics)

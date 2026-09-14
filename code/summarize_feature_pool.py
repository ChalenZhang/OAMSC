"""Reduce local feature pools to per-style moments, without copying samples.

Input files are <Canonical-Style>.npy matrices of shape (examples, channels).
The public data format contains counts, means, and population variances only.
Aggregation does not itself establish license compliance or non-invertibility.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def moments(features, chunk_size=1024):
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("Expected a nonempty examples-by-channels matrix.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    count = 0
    mean = np.zeros(features.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    # Merge batch moments using the parallel form of Welford's algorithm.
    for start in range(0, len(features), chunk_size):
        batch = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("Non-finite feature values are not supported.")
        size = len(batch)
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        total = count + size
        m2 += batch_m2 + delta ** 2 * count * size / total
        mean += delta * size / total
        count = total
    return count, mean, np.maximum(m2 / count, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--style-names", type=Path, required=True)
    parser.add_argument("--extractor", required=True, help="Public extractor identifier, not a path.")
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.min_count < 2:
        parser.error("--min-count must be at least 2")
    if not args.extractor or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in args.extractor):
        parser.error("Use an English identifier for --extractor, without paths or spaces.")
    styles = [line.strip() for line in args.style_names.read_text().splitlines() if line.strip()]
    if not styles or len(set(styles)) != len(styles):
        parser.error("Provide a nonempty list of unique canonical style names.")
    records = []
    channels = None
    for style in styles:
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in style):
            parser.error("Use canonical English style names without paths.")
        features = np.load(args.input_dir / (style + ".npy"), mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or len(features) < args.min_count:
            parser.error("Each style needs an examples-by-channels matrix meeting --min-count.")
        count, mean, variance = moments(features)
        if channels is not None and len(mean) != channels:
            parser.error("Feature dimensions must agree across styles.")
        channels = len(mean)
        records.append({"style": style, "count": count, "mean": mean.tolist(),
                        "population_variance": variance.tolist()})
    result = {"schema_version": 1, "extractor": args.extractor,
              "aggregation": "per-style channel moments over local pooled features",
              "feature_dimension": channels, "styles": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print("Summarized {} styles. No per-example features were exported.".format(len(styles)))


if __name__ == "__main__":
    main()

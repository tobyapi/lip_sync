#!/usr/bin/env python3
"""Train or initialize a tiny diagonal GMM from CSV features.

This intentionally avoids scikit-learn. It provides a documented lightweight
initializer: rows are grouped by label, optionally partitioned into deterministic
mixtures, and each mixture exports diagonal means/variances. Real accuracy still
depends on labeled real-audio features from the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train a small diagonal GMM from CSV features.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with label and feature columns")
    parser.add_argument("--out", required=True, type=Path, help="JSON output path")
    parser.add_argument("--label-column", default="label", help="Class label column")
    parser.add_argument("--feature-prefix", default="feature_", help="Prefix for feature columns")
    parser.add_argument("--mixtures", type=int, default=1, help="Mixtures per class")
    parser.add_argument("--variance-floor", type=float, default=1.0e-4, help="Minimum diagonal variance")
    return parser.parse_args(argv)


def read_rows(path, label_column, feature_prefix):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        feature_columns = [name for name in reader.fieldnames or [] if name.startswith(feature_prefix)]
        if not feature_columns:
            raise ValueError(f"no feature columns with prefix {feature_prefix!r}")
        groups = {}
        for row in reader:
            label = row.get(label_column, "").strip()
            if not label:
                continue
            features = [float(row[name]) for name in feature_columns]
            if not all(math.isfinite(value) for value in features):
                continue
            groups.setdefault(label, []).append(features)
    if not groups:
        raise ValueError("no labeled finite feature rows found")
    return feature_columns, groups


def mean_variance(rows, variance_floor):
    count = len(rows)
    width = len(rows[0])
    means = [sum(row[index] for row in rows) / count for index in range(width)]
    variances = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in rows) / max(1, count - 1)
        variances.append(max(variance, variance_floor))
    return means, variances


def train(groups, mixtures, variance_floor):
    labels = sorted(groups)
    model = {
        "labels": labels,
        "num_classes": len(labels),
        "num_mixtures": mixtures,
        "means": [],
        "inv_vars": [],
        "log_weights": [],
        "class_priors": [],
    }
    total_rows = sum(len(rows) for rows in groups.values())
    for label in labels:
        rows = groups[label]
        model["class_priors"].append(math.log(len(rows) / total_rows))
        for mixture_index in range(mixtures):
            shard = rows[mixture_index::mixtures] or rows
            means, variances = mean_variance(shard, variance_floor)
            model["means"].extend(means)
            model["inv_vars"].extend(1.0 / variance for variance in variances)
            model["log_weights"].append(-math.log(mixtures))
    return model


def main(argv):
    try:
        args = parse_args(argv)
        if args.mixtures < 1:
            raise ValueError("mixtures must be at least 1")
        feature_columns, groups = read_rows(args.input, args.label_column, args.feature_prefix)
        model = train(groups, args.mixtures, args.variance_floor)
        model["feature_columns"] = feature_columns
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

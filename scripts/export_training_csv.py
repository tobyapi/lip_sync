#!/usr/bin/env python3
"""Export labeled lip_sync training features from real WAV clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import evaluate_dataset


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame 16-band and 31-dim training features from labeled WAV clips."
    )
    parser.add_argument("--library", required=True, type=Path, help="Path to compiled lip_sync native library")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to testdata/real_audio dataset")
    parser.add_argument("--out", required=True, type=Path, help="Output training CSV path")
    parser.add_argument("--chunk-ms", type=float, default=20.0, help="Processing chunk size in milliseconds")
    parser.add_argument("--center-region", type=float, default=0.6, help="Center fraction of sustained vowel clips to mark evaluable")
    parser.add_argument("--evaluated-only", action="store_true", help="Write only rows where eval_frame is true")
    parser.add_argument("--singing", action="store_true", help="Enable singing-mode analyzer flag")
    parser.add_argument("--tiny-nn", action="store_true", help="Enable optional tiny NN analyzer flag")
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Enable the trained 16-band spectral GMM path; placeholder fallback reports gmm_model_kind=1",
    )
    parser.add_argument("--no-robust-loudness", dest="robust_loudness", action="store_false", help="Disable robust loudness flag")
    parser.set_defaults(robust_loudness=True)
    parser.add_argument("--metadata-weight", type=float, default=0.0, help="Metadata weight passed to options")
    parser.add_argument("--smoothing", type=float, default=0.18, help="Smoothing passed to options")
    parser.add_argument("--loudness-adaptation", type=float, default=0.07, help="Loudness adaptation passed to options")
    args = parser.parse_args(argv)
    if args.chunk_ms <= 0.0:
        raise evaluate_dataset.EvalError("chunk-ms must be greater than zero")
    return args


def training_fieldnames() -> list[str]:
    return [
        "label",
        "file",
        "time_seconds",
        "sample_rate",
        "eval_frame",
        "best_class",
        "raw_best_vowel",
        "classifier_kind",
        "band_feature_space",
        "feature_vector_space",
    ] + [
        f"band_{index:02d}" for index in range(evaluate_dataset.NUM_BANDS)
    ] + [
        f"feature_{index:02d}" for index in range(evaluate_dataset.FEATURE_VECTOR_LEN)
    ]


def write_training_csv(rows: list[dict[str, object]], out_path: Path, evaluated_only: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = training_fieldnames()
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if evaluated_only and not row.get("eval_frame"):
                continue
            output = {
                "label": row["label"],
                "file": row["file"],
                "time_seconds": row["time_seconds"],
                "sample_rate": row["sample_rate"],
                "eval_frame": row["eval_frame"],
                "best_class": row["best_class"],
                "raw_best_vowel": row["raw_best_vowel"],
                "classifier_kind": row["classifier_kind"],
                "band_feature_space": row["band_feature_space"],
                "feature_vector_space": row["feature_vector_space"],
            }
            for index, value in enumerate(row["band_features"]):
                output[f"band_{index:02d}"] = value
            for index, value in enumerate(row["feature_vector"]):
                output[f"feature_{index:02d}"] = value
            writer.writerow(output)
            written += 1
    return written


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        library = evaluate_dataset.load_library(args.library)
        labeled_files = evaluate_dataset.find_labeled_wavs(args.dataset)
        rows = []
        for path, label in labeled_files:
            rows.extend(evaluate_dataset.process_file(library, path, label, args))
        written = write_training_csv(rows, args.out, args.evaluated_only)
        print(f"wrote {written} rows to {args.out}")
        return 0
    except evaluate_dataset.EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

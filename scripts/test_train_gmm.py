#!/usr/bin/env python3
"""Unit tests for train_gmm CSV loading."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("train_gmm.py")
spec = importlib.util.spec_from_file_location("train_gmm", SCRIPT)
train_gmm = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train_gmm)


def write_band_training_csv(path: Path):
    fieldnames = ["label", "eval_frame"] + [f"band_{index:02d}" for index in range(16)] + ["feature_00"]
    rows = [
        ("O", 0.5),
        ("U", 0.4),
        ("A", 0.1),
        ("REST", 9.9),
        ("E", 0.3),
        ("I", 0.2),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, base in rows:
            row = {"label": label, "eval_frame": "True", "feature_00": base + 100.0}
            for index in range(16):
                row[f"band_{index:02d}"] = base + index * 0.01
            writer.writerow(row)


class TrainingCsvTests(unittest.TestCase):
    def test_reads_band_features_from_export_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["label", "eval_frame", "band_00", "band_01", "feature_00"],
                )
                writer.writeheader()
                writer.writerow(
                    {"label": "A", "eval_frame": "True", "band_00": "0.1", "band_01": "0.2", "feature_00": "9.9"}
                )
                writer.writerow(
                    {"label": "I", "eval_frame": "False", "band_00": "0.3", "band_01": "0.4", "feature_00": "8.8"}
                )

            columns, groups = train_gmm.read_rows(path, "label", "band_", evaluated_only=True)

        self.assertEqual(columns, ["band_00", "band_01"])
        self.assertEqual(sorted(groups), ["A"])
        self.assertEqual(groups["A"], [[0.1, 0.2]])

    def test_band_training_uses_vowel_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            write_band_training_csv(path)
            _, groups = train_gmm.read_rows(path, "label", "band_")
            model = train_gmm.train(groups, mixtures=1, variance_floor=1.0e-4, label_order=train_gmm.VOWEL_LABEL_ORDER)

        self.assertEqual(model["labels"], ["A", "I", "U", "E", "O"])
        self.assertEqual(model["num_classes"], 5)

    def test_feature_set_overrides_prefix(self):
        args = train_gmm.parse_args(
            [
                "--input",
                "training.csv",
                "--out",
                "gmm.json",
                "--feature-set",
                "band",
            ]
        )
        self.assertEqual(args.feature_prefix, "band_")
        self.assertEqual(args.label_order, ["A", "I", "U", "E", "O"])

    def test_feature_set_band_writes_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "training.csv"
            output_path = directory / "gmm.json"
            write_band_training_csv(input_path)

            result = train_gmm.main(
                [
                    "--input",
                    str(input_path),
                    "--out",
                    str(output_path),
                    "--feature-set",
                    "band",
                ]
            )
            model = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(model["feature_prefix"], "band_")
        self.assertEqual(model["labels"], ["A", "I", "U", "E", "O"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for train_gmm CSV loading."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("train_gmm.py")
spec = importlib.util.spec_from_file_location("train_gmm", SCRIPT)
train_gmm = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train_gmm)


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
                writer.writerow({"label": "A", "eval_frame": "True", "band_00": "0.1", "band_01": "0.2", "feature_00": "9.9"})
                writer.writerow({"label": "I", "eval_frame": "False", "band_00": "0.3", "band_01": "0.4", "feature_00": "8.8"})

            columns, groups = train_gmm.read_rows(path, "label", "band_", evaluated_only=True)

        self.assertEqual(columns, ["band_00", "band_01"])
        self.assertEqual(sorted(groups), ["A"])
        self.assertEqual(groups["A"], [[0.1, 0.2]])

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


if __name__ == "__main__":
    unittest.main()
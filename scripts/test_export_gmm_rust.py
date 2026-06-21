#!/usr/bin/env python3
"""Unit tests for export_gmm_rust.py."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("export_gmm_rust.py")
spec = importlib.util.spec_from_file_location("export_gmm_rust", SCRIPT)
export_gmm_rust = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(export_gmm_rust)


def sample_band_model():
    labels = ["A", "I", "U", "E", "O"]
    num_classes = len(labels)
    num_mixtures = 1
    num_features = 16
    return {
        "labels": labels,
        "num_classes": num_classes,
        "num_mixtures": num_mixtures,
        "feature_prefix": "band_",
        "feature_columns": [f"band_{index:02d}" for index in range(num_features)],
        "means": [index * 0.01 for index in range(num_classes * num_mixtures * num_features)],
        "inv_vars": [1.0 for _ in range(num_classes * num_mixtures * num_features)],
        "log_weights": [0.0 for _ in range(num_classes * num_mixtures)],
        "class_priors": [0.0 for _ in range(num_classes)],
    }


class ExportGmmRustTests(unittest.TestCase):
    def test_exports_band_model_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "gmm.json"
            output_path = directory / "trained_band_gmm.rs"
            input_path.write_text(json.dumps(sample_band_model()), encoding="utf-8")

            result = export_gmm_rust.main(["--input", str(input_path), "--out", str(output_path)])
            output = output_path.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertIn('TRAINED_BAND_GMM_CLASSES: [&str; 5]', output)
        self.assertIn('TRAINED_BAND_GMM_FEATURES: usize = 16', output)
        self.assertIn('TRAINED_BAND_GMM_MEANS: [f32; 80]', output)
        self.assertIn('GMM_MODEL_KIND_TRAINED: u32 = 2', output)
        self.assertIn('pub fn trained_band_vowel_gmm_model_kind() -> u32', output)
        self.assertIn('pub fn trained_band_vowel_gmm() -> DiagonalGmm', output)

    def test_rejects_non_band_feature_prefix(self):
        model = sample_band_model()
        model["feature_prefix"] = "feature_"
        with self.assertRaises(export_gmm_rust.ExportError):
            export_gmm_rust.validate_model(model)


if __name__ == "__main__":
    unittest.main()

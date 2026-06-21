#!/usr/bin/env python3
"""Unit tests for evaluate_dataset helper logic."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("evaluate_dataset.py")
spec = importlib.util.spec_from_file_location("evaluate_dataset", SCRIPT)
evaluate_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evaluate_dataset)


class LabelParsingTests(unittest.TestCase):
    def test_vowel_filename_label(self):
        root = Path("testdata/real_audio")
        path = root / "vowels" / "speaker01" / "normal" / "a_01.wav"
        self.assertEqual(evaluate_dataset.label_from_path(path, root), "A")

    def test_non_vowel_directory_label(self):
        root = Path("testdata/real_audio")
        path = root / "non_vowels" / "fricative" / "clip01.wav"
        self.assertEqual(evaluate_dataset.label_from_path(path, root), "FRICATIVE")

    def test_unknown_label_is_rejected(self):
        root = Path("testdata/real_audio")
        path = root / "vowels" / "speaker01" / "normal" / "bad_01.wav"
        with self.assertRaises(evaluate_dataset.EvalError):
            evaluate_dataset.label_from_path(path, root)


class SummaryMathTests(unittest.TestCase):
    def test_summary_metrics(self):
        rows = [
            {
                "file": "a.wav",
                "time_seconds": 0.00,
                "label": "A",
                "best_class": "A",
                "top2": "A|I",
                "jaw_open": 0.5,
                "entropy": 0.2,
                "eval_frame": True,
            },
            {
                "file": "a.wav",
                "time_seconds": 0.02,
                "label": "A",
                "best_class": "I",
                "top2": "I|A",
                "jaw_open": 0.7,
                "entropy": 0.4,
                "eval_frame": True,
            },
            {
                "file": "silence.wav",
                "time_seconds": 0.00,
                "label": "REST",
                "best_class": "REST",
                "top2": "REST|OTHER",
                "jaw_open": 0.0,
                "entropy": 0.1,
                "eval_frame": True,
            },
            {
                "file": "fricative.wav",
                "time_seconds": 0.00,
                "label": "FRICATIVE",
                "best_class": "FRICATIVE",
                "top2": "FRICATIVE|OTHER",
                "jaw_open": 0.2,
                "entropy": 0.3,
                "eval_frame": True,
            },
        ]
        summary = evaluate_dataset.compute_summary(rows)
        self.assertEqual(summary["vowel_top1_accuracy"], 0.5)
        self.assertEqual(summary["vowel_top2_accuracy"], 1.0)
        self.assertEqual(summary["rest_rejection_accuracy"], 1.0)
        self.assertEqual(summary["fricative_detection_accuracy"], 1.0)
        self.assertAlmostEqual(summary["average_jaw_open"], 0.35)


class FeatureColumnTests(unittest.TestCase):
    def test_write_outputs_includes_gmm_model_kind(self):
        posterior = [0.0 for _ in evaluate_dataset.CLASS_NAMES]
        posterior[evaluate_dataset.CLASS_TO_INDEX["A"]] = 1.0
        rows = [
            {
                "file": "a.wav",
                "time_seconds": 0.0,
                "sample_rate": 16_000,
                "label": "A",
                "best_class": "A",
                "top2": "A|I",
                "eval_frame": True,
                "jaw_open": 0.5,
                "vowel_confidence": 1.0,
                "f1_hz": 0.0,
                "f2_hz": 0.0,
                "entropy": 0.0,
                "classifier_kind": evaluate_dataset.LIPSYNC_FLAG_GMM,
                "gmm_model_kind": 1,
                "band_feature_space": evaluate_dataset.NUM_BANDS,
                "feature_vector_space": evaluate_dataset.FEATURE_VECTOR_LEN,
                "posterior": posterior,
                "vowel_scores": [1.0, 0.0, 0.0, 0.0, 0.0],
                "band_features": [0.0 for _ in range(evaluate_dataset.NUM_BANDS)],
                "feature_vector": [0.0 for _ in range(evaluate_dataset.FEATURE_VECTOR_LEN)],
                "activity": 1.0,
                "rms": 0.1,
                "high_ratio": 0.0,
                "zcr": 0.0,
                "flatness": 0.0,
                "compression_likelihood": 0.0,
                "raw_best_vowel": "A",
            }
        ]
        summary = evaluate_dataset.compute_summary(rows)
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            evaluate_dataset.write_outputs(rows, summary, out_dir)
            with (out_dir / "frames.csv").open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))

        self.assertEqual(summary["gmm_model_kind"], 1)
        self.assertEqual(summary["gmm_model_kind_counts"], {"1": 1})
        self.assertEqual(output_rows[0]["gmm_model_kind"], "1")
    def test_feature_column_dimensions_are_stable(self):
        self.assertEqual(evaluate_dataset.NUM_BANDS, 16)
        self.assertEqual(evaluate_dataset.FEATURE_VECTOR_LEN, 31)


class OptionFlagTests(unittest.TestCase):
    def parse(self, *extra_args):
        return evaluate_dataset.parse_args(
            [
                "--library",
                "target/release/lip_sync.dll",
                "--dataset",
                "testdata/real_audio",
                "--out",
                "target/lipsync_eval",
                *extra_args,
            ]
        )

    def test_gmm_flag_sets_lipsync_flag_gmm(self):
        args = self.parse("--gmm")
        options = evaluate_dataset.build_options(16_000, args)
        self.assertTrue(options.flags & evaluate_dataset.LIPSYNC_FLAG_GMM)

    def test_gmm_flag_defaults_off(self):
        args = self.parse()
        options = evaluate_dataset.build_options(16_000, args)
        self.assertFalse(options.flags & evaluate_dataset.LIPSYNC_FLAG_GMM)


if __name__ == "__main__":
    unittest.main()

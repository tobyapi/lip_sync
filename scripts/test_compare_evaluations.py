#!/usr/bin/env python3
"""Unit tests for compare_evaluations.py."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("compare_evaluations.py")
spec = importlib.util.spec_from_file_location("compare_evaluations", SCRIPT)
compare_evaluations = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(compare_evaluations)


CLASS_NAMES = compare_evaluations.CLASS_NAMES


def frame(label, best_class, top2, raw_best_vowel, time_seconds):
    return {
        "file": f"{label.lower()}.wav",
        "time_seconds": f"{time_seconds:.2f}",
        "label": label,
        "best_class": best_class,
        "top2": top2,
        "eval_frame": "True",
        "raw_best_vowel": raw_best_vowel,
    }


def confusion_from_frames(frames):
    matrix = {label: {predicted: 0 for predicted in CLASS_NAMES} for label in CLASS_NAMES}
    for row in frames:
        matrix[row["label"]][row["best_class"]] += 1
    return matrix


def write_eval_dir(path: Path, frames, summary):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with (path / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "time_seconds", "label", "best_class", "top2", "eval_frame", "raw_best_vowel"],
        )
        writer.writeheader()
        writer.writerows(frames)

    matrix = confusion_from_frames(frames)
    with (path / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted", *CLASS_NAMES])
        for expected in CLASS_NAMES:
            writer.writerow([expected, *[matrix[expected][predicted] for predicted in CLASS_NAMES]])


class CompareEvaluationTests(unittest.TestCase):
    def test_metric_calculation_and_outputs(self):
        baseline_frames = [
            frame("A", "OTHER", "OTHER|A", "A", 0.00),
            frame("I", "E", "E|I", "E", 0.02),
            frame("U", "U", "U|O", "U", 0.04),
            frame("O", "O", "O|A", "A", 0.06),
            frame("E", "REST", "REST|OTHER", "A", 0.08),
            frame("REST", "REST", "REST|OTHER", "", 0.10),
        ]
        candidate_frames = [
            frame("A", "A", "A|O", "A", 0.00),
            frame("I", "I", "I|E", "I", 0.02),
            frame("U", "O", "O|U", "O", 0.04),
            frame("O", "O", "O|A", "O", 0.06),
            frame("E", "E", "E|A", "A", 0.08),
            frame("REST", "REST", "REST|OTHER", "", 0.10),
        ]
        baseline_summary = {
            "vowel_top1_accuracy": 0.4,
            "vowel_top2_accuracy": 0.8,
            "class_switches_per_second": 2.0,
        }
        candidate_summary = {
            "vowel_top1_accuracy": 0.8,
            "vowel_top2_accuracy": 1.0,
            "class_switches_per_second": 1.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir = root / "baseline"
            candidate_dir = root / "candidate"
            out_dir = root / "compare"
            write_eval_dir(baseline_dir, baseline_frames, baseline_summary)
            write_eval_dir(candidate_dir, candidate_frames, candidate_summary)

            with contextlib.redirect_stdout(io.StringIO()):
                result = compare_evaluations.main(
                    ["--baseline", str(baseline_dir), "--candidate", str(candidate_dir), "--out", str(out_dir)]
                )
            report = json.loads((out_dir / "compare_summary.json").read_text(encoding="utf-8"))
            with (out_dir / "per_label_metrics.csv").open(newline="", encoding="utf-8") as handle:
                per_label_rows = list(csv.DictReader(handle))
            with (out_dir / "confusion_delta.csv").open(newline="", encoding="utf-8") as handle:
                confusion_rows = list(csv.DictReader(handle))
            with (out_dir / "error_breakdown.csv").open(newline="", encoding="utf-8") as handle:
                error_rows = list(csv.DictReader(handle))

        self.assertEqual(result, 0)
        self.assertAlmostEqual(report["delta"]["vowel_top1_accuracy"], 0.4)
        self.assertAlmostEqual(report["delta"]["vowel_top2_accuracy"], 0.2)
        self.assertAlmostEqual(report["delta"]["class_switches_per_second"], -0.5)
        self.assertAlmostEqual(report["baseline"]["raw_best_vowel_accuracy"], 0.4)
        self.assertAlmostEqual(report["candidate"]["final_best_class_accuracy"], 5 / 6)
        self.assertEqual(report["baseline"]["raw_correct_but_final_wrong_count"], 1)
        self.assertEqual(report["candidate"]["raw_wrong_but_final_correct_count"], 1)
        self.assertAlmostEqual(report["baseline"]["other_absorption_rate"], 0.2)
        self.assertAlmostEqual(report["baseline"]["rest_absorption_rate"], 0.2)
        self.assertEqual(report["delta"]["pair_confusions"]["I_E"], -1)
        self.assertEqual(report["delta"]["pair_confusions"]["U_O"], 1)

        per_label = {row["label"]: row for row in per_label_rows}
        self.assertEqual(per_label["I"]["baseline_recall"], "0.0")
        self.assertEqual(per_label["I"]["candidate_recall"], "1.0")

        ie_rows = [row for row in confusion_rows if row["expected"] == "I" and row["predicted"] == "E"]
        self.assertEqual(ie_rows[0]["delta"], "-1")

        candidate_d = [row for row in error_rows if row["evaluation"] == "candidate" and row["category"] == "D"]
        self.assertEqual(candidate_d[0]["count"], "0")

    def test_compute_metrics_categorizes_errors(self):
        rows = [
            frame("A", "OTHER", "OTHER|A", "A", 0.00),
            frame("E", "REST", "REST|OTHER", "A", 0.02),
        ]
        evaluation = {
            "path": "synthetic",
            "summary": {"vowel_top1_accuracy": 0.0, "vowel_top2_accuracy": 0.5, "class_switches_per_second": 0.0},
            "confusion": confusion_from_frames(rows),
            "frames": rows,
        }

        metrics = compare_evaluations.compute_metrics(evaluation)

        self.assertEqual(metrics["error_categories"]["A"], 1)
        self.assertEqual(metrics["error_categories"]["B"], 1)
        self.assertEqual(metrics["error_categories"]["C"], 1)
        self.assertEqual(metrics["error_categories"]["D"], 2)


if __name__ == "__main__":
    unittest.main()
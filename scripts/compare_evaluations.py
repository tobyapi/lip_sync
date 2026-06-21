#!/usr/bin/env python3
"""Compare two lip_sync evaluation output directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

CLASS_NAMES = ["REST", "CLOSED", "A", "I", "U", "E", "O", "FRICATIVE", "OTHER"]
VOWEL_NAMES = ["A", "I", "U", "E", "O"]
CONFUSION_PAIRS = [("I", "E"), ("U", "O"), ("A", "O")]
ERROR_CATEGORIES = {
    "A": "raw_best_vowel == label but best_class != label",
    "B": "label in top2 but best_class != label",
    "C": "raw_best_vowel != label and label not in top2",
    "D": "predicted REST/OTHER for vowel labels",
}


class CompareError(Exception):
    """User-facing comparison error."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate lip_sync evaluation outputs.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline evaluation output directory")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate evaluation output directory")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for comparison reports")
    return parser.parse_args(argv)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def read_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise CompareError(f"missing evaluation file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompareError(f"invalid JSON file: {path}: {exc}") from exc


def read_frames(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise CompareError(f"missing evaluation file: {path}") from exc


def empty_confusion() -> dict[str, dict[str, int]]:
    return {label: {predicted: 0 for predicted in CLASS_NAMES} for label in CLASS_NAMES}


def read_confusion(path: Path) -> dict[str, dict[str, int]]:
    matrix = empty_confusion()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or len(header) < 2:
                raise CompareError(f"invalid confusion matrix header: {path}")
            predicted_labels = header[1:]
            for row in reader:
                if not row:
                    continue
                expected = row[0]
                if expected not in matrix:
                    continue
                for predicted, raw_value in zip(predicted_labels, row[1:]):
                    if predicted in matrix[expected]:
                        matrix[expected][predicted] = int(raw_value or 0)
    except OSError as exc:
        raise CompareError(f"missing evaluation file: {path}") from exc
    except ValueError as exc:
        raise CompareError(f"invalid confusion matrix value: {path}: {exc}") from exc
    return matrix


def load_evaluation(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "summary": read_json(path / "summary.json"),
        "confusion": read_confusion(path / "confusion_matrix.csv"),
        "frames": read_frames(path / "frames.csv"),
    }


def top2_values(row: dict[str, str]) -> set[str]:
    return {value.strip() for value in row.get("top2", "").split("|") if value.strip()}


def evaluated_rows(frames: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in frames if parse_bool(row.get("eval_frame", ""))]


def confusion_pair_count(matrix: dict[str, dict[str, int]], first: str, second: str) -> int:
    return matrix[first][second] + matrix[second][first]


def per_label_stats(rows: list[dict[str, str]], label: str) -> dict[str, float | int | None]:
    support = sum(1 for row in rows if row.get("label") == label)
    predicted = sum(1 for row in rows if row.get("best_class") == label)
    true_positive = sum(1 for row in rows if row.get("label") == label and row.get("best_class") == label)
    return {
        "support": support,
        "predicted": predicted,
        "true_positive": true_positive,
        "recall": ratio(true_positive, support),
        "precision": ratio(true_positive, predicted),
    }


def error_category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {key: 0 for key in ERROR_CATEGORIES}
    for row in rows:
        label = row.get("label", "")
        if label not in VOWEL_NAMES:
            continue
        best_class = row.get("best_class", "")
        raw_best = row.get("raw_best_vowel", "")
        top2 = top2_values(row)
        if raw_best == label and best_class != label:
            counts["A"] += 1
        if label in top2 and best_class != label:
            counts["B"] += 1
        if raw_best != label and label not in top2:
            counts["C"] += 1
        if best_class in {"REST", "OTHER"}:
            counts["D"] += 1
    return counts



def gmm_model_kind_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw_value = row.get("gmm_model_kind", "")
        if str(raw_value).strip() == "":
            continue
        try:
            key = str(int(raw_value))
        except ValueError:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def single_gmm_model_kind(counts: dict[str, int]) -> int | None:
    if len(counts) != 1:
        return None
    return int(next(iter(counts)))

def compute_metrics(evaluation: dict[str, object]) -> dict[str, object]:
    summary = evaluation["summary"]
    confusion = evaluation["confusion"]
    rows = evaluated_rows(evaluation["frames"])
    vowel_rows = [row for row in rows if row.get("label") in VOWEL_NAMES]

    raw_correct = sum(1 for row in vowel_rows if row.get("raw_best_vowel") == row.get("label"))
    final_vowel_correct = sum(1 for row in vowel_rows if row.get("best_class") == row.get("label"))
    final_all_correct = sum(1 for row in rows if row.get("best_class") == row.get("label"))
    raw_correct_final_wrong = sum(
        1 for row in vowel_rows if row.get("raw_best_vowel") == row.get("label") and row.get("best_class") != row.get("label")
    )
    raw_wrong_final_correct = sum(
        1 for row in vowel_rows if row.get("raw_best_vowel") != row.get("label") and row.get("best_class") == row.get("label")
    )
    other_absorbed = sum(1 for row in vowel_rows if row.get("best_class") == "OTHER")
    rest_absorbed = sum(1 for row in vowel_rows if row.get("best_class") == "REST")

    per_label = {label: per_label_stats(rows, label) for label in VOWEL_NAMES}
    pair_confusions = {
        f"{first}_{second}": confusion_pair_count(confusion, first, second) for first, second in CONFUSION_PAIRS
    }
    gmm_counts = gmm_model_kind_counts(rows)

    return {
        "path": evaluation["path"],
        "evaluated_frames": len(rows),
        "vowel_frames": len(vowel_rows),
        "vowel_top1_accuracy": parse_float(summary.get("vowel_top1_accuracy")),
        "vowel_top2_accuracy": parse_float(summary.get("vowel_top2_accuracy")),
        "gmm_model_kind": single_gmm_model_kind(gmm_counts),
        "gmm_model_kind_counts": gmm_counts,
        "class_switches_per_second": parse_float(summary.get("class_switches_per_second")),
        "raw_best_vowel_accuracy": ratio(raw_correct, len(vowel_rows)),
        "final_best_class_accuracy": ratio(final_all_correct, len(rows)),
        "final_vowel_accuracy": ratio(final_vowel_correct, len(vowel_rows)),
        "raw_correct_but_final_wrong_count": raw_correct_final_wrong,
        "raw_wrong_but_final_correct_count": raw_wrong_final_correct,
        "other_absorption_rate": ratio(other_absorbed, len(vowel_rows)),
        "rest_absorption_rate": ratio(rest_absorbed, len(vowel_rows)),
        "per_label": per_label,
        "pair_confusions": pair_confusions,
        "error_categories": error_category_counts(rows),
    }


def compare_metrics(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    metric_keys = [
        "vowel_top1_accuracy",
        "vowel_top2_accuracy",
        "raw_best_vowel_accuracy",
        "final_best_class_accuracy",
        "final_vowel_accuracy",
        "other_absorption_rate",
        "rest_absorption_rate",
        "class_switches_per_second",
    ]
    count_keys = [
        "raw_correct_but_final_wrong_count",
        "raw_wrong_but_final_correct_count",
    ]
    metric_deltas = {key: delta(candidate.get(key), baseline.get(key)) for key in metric_keys}
    count_deltas = {key: int(candidate.get(key, 0)) - int(baseline.get(key, 0)) for key in count_keys}
    pair_deltas = {
        key: int(candidate["pair_confusions"].get(key, 0)) - int(baseline["pair_confusions"].get(key, 0))
        for key in sorted(set(baseline["pair_confusions"]) | set(candidate["pair_confusions"]))
    }
    category_deltas = {
        key: int(candidate["error_categories"].get(key, 0)) - int(baseline["error_categories"].get(key, 0))
        for key in ERROR_CATEGORIES
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            **metric_deltas,
            **count_deltas,
            "pair_confusions": pair_deltas,
            "error_categories": category_deltas,
        },
    }


def write_compare_summary(report: dict[str, object], out_dir: Path) -> None:
    with (out_dir / "compare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_confusion_delta(
    baseline: dict[str, dict[str, int]], candidate: dict[str, dict[str, int]], out_dir: Path
) -> None:
    with (out_dir / "confusion_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expected", "predicted", "baseline", "candidate", "delta"])
        writer.writeheader()
        for expected in CLASS_NAMES:
            for predicted in CLASS_NAMES:
                baseline_count = baseline[expected][predicted]
                candidate_count = candidate[expected][predicted]
                writer.writerow(
                    {
                        "expected": expected,
                        "predicted": predicted,
                        "baseline": baseline_count,
                        "candidate": candidate_count,
                        "delta": candidate_count - baseline_count,
                    }
                )


def write_per_label_metrics(baseline: dict[str, object], candidate: dict[str, object], out_dir: Path) -> None:
    fieldnames = [
        "label",
        "baseline_support",
        "candidate_support",
        "baseline_predicted",
        "candidate_predicted",
        "baseline_recall",
        "candidate_recall",
        "recall_delta",
        "baseline_precision",
        "candidate_precision",
        "precision_delta",
    ]
    with (out_dir / "per_label_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in VOWEL_NAMES:
            base = baseline["per_label"][label]
            cand = candidate["per_label"][label]
            writer.writerow(
                {
                    "label": label,
                    "baseline_support": base["support"],
                    "candidate_support": cand["support"],
                    "baseline_predicted": base["predicted"],
                    "candidate_predicted": cand["predicted"],
                    "baseline_recall": base["recall"],
                    "candidate_recall": cand["recall"],
                    "recall_delta": delta(cand["recall"], base["recall"]),
                    "baseline_precision": base["precision"],
                    "candidate_precision": cand["precision"],
                    "precision_delta": delta(cand["precision"], base["precision"]),
                }
            )


def write_error_breakdown(baseline: dict[str, object], candidate: dict[str, object], out_dir: Path) -> None:
    fieldnames = ["evaluation", "category", "description", "count", "vowel_frames", "rate", "delta_vs_baseline"]
    with (out_dir / "error_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation_name, metrics in [("baseline", baseline), ("candidate", candidate)]:
            vowel_frames = int(metrics["vowel_frames"])
            for category, description in ERROR_CATEGORIES.items():
                count = int(metrics["error_categories"].get(category, 0))
                baseline_count = int(baseline["error_categories"].get(category, 0))
                writer.writerow(
                    {
                        "evaluation": evaluation_name,
                        "category": category,
                        "description": description,
                        "count": count,
                        "vowel_frames": vowel_frames,
                        "rate": ratio(count, vowel_frames),
                        "delta_vs_baseline": count - baseline_count if evaluation_name == "candidate" else 0,
                    }
                )


def write_outputs(
    report: dict[str, object],
    baseline_eval: dict[str, object],
    candidate_eval: dict[str, object],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_compare_summary(report, out_dir)
    write_confusion_delta(baseline_eval["confusion"], candidate_eval["confusion"], out_dir)
    write_per_label_metrics(report["baseline"], report["candidate"], out_dir)
    write_error_breakdown(report["baseline"], report["candidate"], out_dir)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        baseline_eval = load_evaluation(args.baseline)
        candidate_eval = load_evaluation(args.candidate)
        baseline_metrics = compute_metrics(baseline_eval)
        candidate_metrics = compute_metrics(candidate_eval)
        report = compare_metrics(baseline_metrics, candidate_metrics)
        write_outputs(report, baseline_eval, candidate_eval, args.out)
        print(json.dumps(report["delta"], indent=2, sort_keys=True))
        return 0
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
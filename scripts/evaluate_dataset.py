#!/usr/bin/env python3
"""Evaluate lip_sync on labeled real WAV clips.

The script intentionally uses the Python standard library only. It loads the
compiled native library with ctypes, scans a simple dataset layout, writes a
per-frame CSV, and writes summary metrics that make classifier changes easier
to compare without claiming accuracy from synthetic tests alone. The GMM path becomes meaningful once a trained 16-band model is exported into src/trained_band_gmm.rs; otherwise it falls back to the placeholder model.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import sys
import wave

CLASS_NAMES = ["REST", "CLOSED", "A", "I", "U", "E", "O", "FRICATIVE", "OTHER"]
VOWEL_NAMES = ["A", "I", "U", "E", "O"]
NUM_BANDS = 16
FEATURE_VECTOR_LEN = 31
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
VOWEL_CLASSES = {"A", "I", "U", "E", "O"}
NON_VOWEL_DIR_LABELS = {
    "silence": "REST",
    "rest": "REST",
    "closed": "CLOSED",
    "fricative": "FRICATIVE",
    "breath": "OTHER",
    "noise": "OTHER",
    "other": "OTHER",
}
FILENAME_LABELS = {
    "a": "A",
    "i": "I",
    "u": "U",
    "e": "E",
    "o": "O",
    "rest": "REST",
    "silence": "REST",
    "closed": "CLOSED",
    "fricative": "FRICATIVE",
    "breath": "OTHER",
    "noise": "OTHER",
    "other": "OTHER",
}
LIPSYNC_FLAG_SINGING_MODE = 1 << 0
LIPSYNC_FLAG_TINY_NN = 1 << 1
LIPSYNC_FLAG_TIMED_CUES = 1 << 2
LIPSYNC_FLAG_ROBUST_LOUDNESS = 1 << 3
LIPSYNC_FLAG_GMM = 1 << 4


class EvalError(Exception):
    """User-facing evaluation error."""


class LipSyncOptions(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("metadata_weight", ctypes.c_float),
        ("smoothing", ctypes.c_float),
        ("loudness_adaptation", ctypes.c_float),
    ]


class LipSyncFrame(ctypes.Structure):
    _fields_ = [
        ("posterior", ctypes.c_float * len(CLASS_NAMES)),
        ("jaw_open", ctypes.c_float),
        ("vowel_confidence", ctypes.c_float),
        ("f1_hz", ctypes.c_float),
        ("f2_hz", ctypes.c_float),
    ]


class LipSyncDebugFrame(ctypes.Structure):
    _fields_ = [
        ("frame", LipSyncFrame),
        ("vowel_scores", ctypes.c_float * len(VOWEL_NAMES)),
        ("normalized_bands", ctypes.c_float * NUM_BANDS),
        ("feature_vector", ctypes.c_float * FEATURE_VECTOR_LEN),
        ("classifier_kind", ctypes.c_uint32),
        ("band_feature_space", ctypes.c_uint32),
        ("feature_vector_space", ctypes.c_uint32),
        ("activity", ctypes.c_float),
        ("rms", ctypes.c_float),
        ("high_ratio", ctypes.c_float),
        ("zero_crossing_rate", ctypes.c_float),
        ("spectral_flatness", ctypes.c_float),
        ("compression_likelihood", ctypes.c_float),
        ("raw_best_vowel", ctypes.c_int32),
    ]


def label_from_path(path: Path, dataset_root: Path) -> str:
    """Infer the expected class from dataset-relative path and filename."""
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        relative = path

    parts = [part.lower() for part in relative.parts]
    stem = path.stem.lower()
    first_token = stem.replace("-", "_").replace(".", "_").split("_")[0]
    if first_token in FILENAME_LABELS:
        label = FILENAME_LABELS[first_token]
        if "vowels" in parts and label not in VOWEL_CLASSES:
            raise EvalError(f"wrong class label for vowel file: {path}")
        return label

    for part in reversed(parts[:-1]):
        if part in NON_VOWEL_DIR_LABELS:
            return NON_VOWEL_DIR_LABELS[part]

    if "vowels" in parts:
        raise EvalError(
            f"wrong class label for vowel file {path}: filename should start with a_, i_, u_, e_, or o_"
        )

    raise EvalError(
        f"wrong class label for {path}: expected filename prefix or non_vowels class directory"
    )


def find_labeled_wavs(dataset_root: Path) -> list[tuple[Path, str]]:
    if not dataset_root.exists():
        raise EvalError(f"dataset directory does not exist: {dataset_root}")

    wavs = []
    for path in sorted(dataset_root.rglob("*.wav")):
        wavs.append((path, label_from_path(path, dataset_root)))

    if not wavs:
        raise EvalError(f"no labeled files found under {dataset_root}")
    return wavs


def _decode_pcm(frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in frames]
    if sample_width == 2:
        count = len(frames) // 2
        return [sample / 32768.0 for sample in struct.unpack("<" + "h" * count, frames)]
    if sample_width == 3:
        samples = []
        for offset in range(0, len(frames), 3):
            raw = frames[offset : offset + 3]
            if len(raw) != 3:
                break
            value = int.from_bytes(raw + (b"\xff" if raw[2] & 0x80 else b"\x00"), "little", signed=True)
            samples.append(value / 8388608.0)
        return samples
    if sample_width == 4:
        count = len(frames) // 4
        return [sample / 2147483648.0 for sample in struct.unpack("<" + "i" * count, frames)]
    raise EvalError(f"unsupported WAV format: sample width {sample_width * 8} bits")


def read_wav_mono(path: Path) -> tuple[int, list[float]]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            if compression != "NONE":
                raise EvalError(f"unsupported WAV format for {path}: compressed WAV type {compression}")
            if channels < 1:
                raise EvalError(f"unsupported WAV format for {path}: no channels")
            if channels > 2:
                raise EvalError(f"unsupported WAV format for {path}: only mono/stereo WAV is supported")
            samples = _decode_pcm(wav.readframes(frame_count), sample_width)
    except wave.Error as exc:
        raise EvalError(f"unsupported WAV format for {path}: {exc}") from exc

    if channels == 1:
        return sample_rate, samples

    mono = []
    for offset in range(0, len(samples) - 1, channels):
        mono.append(sum(samples[offset : offset + channels]) / channels)
    return sample_rate, mono


def load_library(path: Path):
    if not path.exists():
        raise EvalError(f"missing native library: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise EvalError(f"missing native library or dependency: {path}: {exc}") from exc

    library.lipsync_create_with_options.argtypes = [LipSyncOptions]
    library.lipsync_create_with_options.restype = ctypes.c_void_p
    library.lipsync_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(LipSyncFrame),
    ]
    library.lipsync_process.restype = ctypes.c_bool
    library.lipsync_process_debug.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(LipSyncDebugFrame),
    ]
    library.lipsync_process_debug.restype = ctypes.c_bool
    library.lipsync_destroy.argtypes = [ctypes.c_void_p]
    library.lipsync_destroy.restype = None
    return library


def build_options(sample_rate: int, args: argparse.Namespace) -> LipSyncOptions:
    flags = 0
    if args.singing:
        flags |= LIPSYNC_FLAG_SINGING_MODE
    if args.tiny_nn:
        flags |= LIPSYNC_FLAG_TINY_NN
    if args.robust_loudness:
        flags |= LIPSYNC_FLAG_ROBUST_LOUDNESS
    if args.gmm:
        flags |= LIPSYNC_FLAG_GMM
    return LipSyncOptions(sample_rate, flags, args.metadata_weight, args.smoothing, args.loudness_adaptation)


def best_class(posterior: list[float]) -> str:
    return CLASS_NAMES[max(range(len(posterior)), key=lambda index: posterior[index])]


def top_k_classes(posterior: list[float], k: int) -> list[str]:
    indices = sorted(range(len(posterior)), key=lambda index: posterior[index], reverse=True)[:k]
    return [CLASS_NAMES[index] for index in indices]


def raw_best_vowel_name(index: int) -> str:
    if 0 <= index < len(VOWEL_NAMES):
        return VOWEL_NAMES[index]
    return ""


def posterior_entropy(posterior: list[float]) -> float:
    entropy = 0.0
    for value in posterior:
        if value > 0.0:
            entropy -= value * math.log(value, 2)
    return entropy


def should_evaluate_frame(label: str, time_seconds: float, duration_seconds: float, center_region: float) -> bool:
    if label not in VOWEL_CLASSES:
        return True
    if duration_seconds <= 0.0:
        return False
    center_region = max(0.0, min(1.0, center_region))
    trim = (1.0 - center_region) * 0.5 * duration_seconds
    return trim <= time_seconds <= duration_seconds - trim


def process_file(library, path: Path, label: str, args: argparse.Namespace) -> list[dict[str, object]]:
    sample_rate, samples = read_wav_mono(path)
    if not samples:
        raise EvalError(f"unsupported WAV format for {path}: no PCM frames")

    chunk_samples = max(1, int(sample_rate * args.chunk_ms / 1000.0))
    duration_seconds = len(samples) / sample_rate
    analyzer = library.lipsync_create_with_options(build_options(sample_rate, args))
    if not analyzer:
        raise EvalError(f"native library failed to create analyzer for {path}")

    rows = []
    try:
        for start in range(0, len(samples), chunk_samples):
            chunk = samples[start : start + chunk_samples]
            array_type = ctypes.c_float * len(chunk)
            debug = LipSyncDebugFrame()
            ok = library.lipsync_process_debug(analyzer, array_type(*chunk), len(chunk), ctypes.byref(debug))
            if not ok:
                raise EvalError(f"native library failed while processing {path} at sample {start}")
            frame = debug.frame
            posterior = [float(frame.posterior[index]) for index in range(len(CLASS_NAMES))]
            vowel_scores = [float(debug.vowel_scores[index]) for index in range(len(VOWEL_NAMES))]
            band_features = [float(debug.normalized_bands[index]) for index in range(NUM_BANDS)]
            feature_vector = [float(debug.feature_vector[index]) for index in range(FEATURE_VECTOR_LEN)]
            time_seconds = start / sample_rate
            rows.append(
                {
                    "file": str(path),
                    "time_seconds": time_seconds,
                    "sample_rate": sample_rate,
                    "label": label,
                    "best_class": best_class(posterior),
                    "top2": "|".join(top_k_classes(posterior, 2)),
                    "jaw_open": float(frame.jaw_open),
                    "vowel_confidence": float(frame.vowel_confidence),
                    "f1_hz": float(frame.f1_hz),
                    "f2_hz": float(frame.f2_hz),
                    "posterior": posterior,
                    "vowel_scores": vowel_scores,
                    "band_features": band_features,
                    "feature_vector": feature_vector,
                    "classifier_kind": int(debug.classifier_kind),
                    "band_feature_space": int(debug.band_feature_space),
                    "feature_vector_space": int(debug.feature_vector_space),
                    "activity": float(debug.activity),
                    "rms": float(debug.rms),
                    "high_ratio": float(debug.high_ratio),
                    "zcr": float(debug.zero_crossing_rate),
                    "flatness": float(debug.spectral_flatness),
                    "compression_likelihood": float(debug.compression_likelihood),
                    "raw_best_vowel": raw_best_vowel_name(int(debug.raw_best_vowel)),
                    "entropy": posterior_entropy(posterior),
                    "eval_frame": should_evaluate_frame(
                        label, time_seconds + (len(chunk) / sample_rate) * 0.5, duration_seconds, args.center_region
                    ),
                }
            )
    finally:
        library.lipsync_destroy(analyzer)
    return rows


def fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    eval_rows = [row for row in rows if row.get("eval_frame")]
    if not eval_rows:
        raise EvalError("no labeled files found with evaluable frames; check clip length and center-region")

    vowel_rows = [row for row in eval_rows if row["label"] in VOWEL_CLASSES]
    rest_rows = [row for row in eval_rows if row["label"] == "REST"]
    fricative_rows = [row for row in eval_rows if row["label"] == "FRICATIVE"]
    closed_rows = [row for row in eval_rows if row["label"] == "CLOSED"]

    def count_best(expected: str, group: list[dict[str, object]]) -> int:
        return sum(1 for row in group if row["best_class"] == expected)

    top1 = sum(1 for row in vowel_rows if row["best_class"] == row["label"])
    top2 = sum(1 for row in vowel_rows if row["label"] in str(row["top2"]).split("|"))
    switches = 0
    total_duration = 0.0
    rows_by_file: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        rows_by_file.setdefault(str(row["file"]), []).append(row)
    for file_rows in rows_by_file.values():
        file_rows = sorted(file_rows, key=lambda row: float(row["time_seconds"]))
        for previous, current in zip(file_rows, file_rows[1:]):
            if previous["best_class"] != current["best_class"]:
                switches += 1
        if len(file_rows) > 1:
            total_duration += float(file_rows[-1]["time_seconds"]) - float(file_rows[0]["time_seconds"])

    confusion = {label: {name: 0 for name in CLASS_NAMES} for label in CLASS_NAMES}
    for row in eval_rows:
        confusion[str(row["label"])][str(row["best_class"])] += 1

    return {
        "files": len(rows_by_file),
        "frames": len(rows),
        "evaluated_frames": len(eval_rows),
        "vowel_frames": len(vowel_rows),
        "vowel_top1_accuracy": fraction(top1, len(vowel_rows)),
        "vowel_top2_accuracy": fraction(top2, len(vowel_rows)),
        "rest_rejection_accuracy": fraction(count_best("REST", rest_rows), len(rest_rows)),
        "fricative_detection_accuracy": fraction(count_best("FRICATIVE", fricative_rows), len(fricative_rows)),
        "closed_detection_accuracy": fraction(count_best("CLOSED", closed_rows), len(closed_rows)),
        "average_jaw_open": sum(float(row["jaw_open"]) for row in eval_rows) / len(eval_rows),
        "class_switches_per_second": fraction(switches, total_duration),
        "mean_posterior_entropy": sum(float(row["entropy"]) for row in eval_rows) / len(eval_rows),
        "confusion_matrix": confusion,
    }


def write_outputs(rows: list[dict[str, object]], summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frames.csv"
    fieldnames = [
        "file",
        "time_seconds",
        "sample_rate",
        "label",
        "best_class",
        "top2",
        "eval_frame",
        "jaw_open",
        "vowel_confidence",
        "f1_hz",
        "f2_hz",
        "entropy",
        "classifier_kind",
        "band_feature_space",
        "feature_vector_space",
    ] + [f"p_{name.lower()}" for name in CLASS_NAMES] + [
        f"vowel_scores_{name.lower()}" for name in VOWEL_NAMES
    ] + [f"band_{index:02d}" for index in range(NUM_BANDS)] + [
        f"feature_{index:02d}" for index in range(FEATURE_VECTOR_LEN)
    ] + [
        "activity",
        "rms",
        "high_ratio",
        "zcr",
        "flatness",
        "compression_likelihood",
        "raw_best_vowel",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fieldnames if key in row}
            for name, value in zip(CLASS_NAMES, row["posterior"]):
                output[f"p_{name.lower()}"] = value
            for name, value in zip(VOWEL_NAMES, row["vowel_scores"]):
                output[f"vowel_scores_{name.lower()}"] = value
            for index, value in enumerate(row["band_features"]):
                output[f"band_{index:02d}"] = value
            for index, value in enumerate(row["feature_vector"]):
                output[f"feature_{index:02d}"] = value
            writer.writerow(output)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected\\predicted"] + CLASS_NAMES)
        matrix = summary["confusion_matrix"]
        for expected in CLASS_NAMES:
            writer.writerow([expected] + [matrix[expected][predicted] for predicted in CLASS_NAMES])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lip_sync on labeled WAV clips.",
        epilog="Note: --gmm evaluates the trained 16-band spectral GMM after src/trained_band_gmm.rs is generated. Empty generated arrays fall back to the placeholder model.",
    )
    parser.add_argument("--library", required=True, type=Path, help="Path to compiled lip_sync native library")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to testdata/real_audio dataset")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for CSV/JSON reports")
    parser.add_argument("--chunk-ms", type=float, default=20.0, help="Processing chunk size in milliseconds")
    parser.add_argument("--center-region", type=float, default=0.6, help="Center fraction of sustained vowel clips to evaluate")
    parser.add_argument("--singing", action="store_true", help="Enable singing-mode analyzer flag")
    parser.add_argument("--tiny-nn", action="store_true", help="Enable optional tiny NN analyzer flag")
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Enable the trained 16-band spectral GMM path, with placeholder fallback when no model is generated",
    )
    parser.add_argument("--no-robust-loudness", dest="robust_loudness", action="store_false", help="Disable robust loudness flag")
    parser.set_defaults(robust_loudness=True)
    parser.add_argument("--metadata-weight", type=float, default=0.0, help="Metadata weight passed to options")
    parser.add_argument("--smoothing", type=float, default=0.18, help="Smoothing passed to options")
    parser.add_argument("--loudness-adaptation", type=float, default=0.07, help="Loudness adaptation passed to options")
    args = parser.parse_args(argv)
    if args.chunk_ms <= 0.0:
        raise EvalError("chunk-ms must be greater than zero")
    return args


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        library = load_library(args.library)
        labeled_files = find_labeled_wavs(args.dataset)
        rows = []
        for path, label in labeled_files:
            rows.extend(process_file(library, path, label, args))
        summary = compute_summary(rows)
        write_outputs(rows, summary, args.out)
        print(json.dumps({key: value for key, value in summary.items() if key != "confusion_matrix"}, indent=2, sort_keys=True))
        return 0
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# Scripts

## Real-audio evaluation

Build the native library, then run the evaluator against a local labeled WAV dataset:

```sh
cargo build --release
python3 scripts/evaluate_dataset.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/lipsync_eval
```

On Windows, pass the DLL path instead:

```sh
python scripts/evaluate_dataset.py \
  --library target/release/lip_sync.dll \
  --dataset testdata/real_audio \
  --out target/lipsync_eval
```

The evaluator writes:

- `frames.csv`: per-frame posterior, best class, jaw opening, confidence, entropy, `band_00..band_15`, and `feature_00..feature_30`.
- `summary.json`: top-1/top-2 vowel accuracy, rest rejection, fricative/closed detection when labels exist, average jaw opening, class switches per second, and posterior entropy.
- `confusion_matrix.csv`: expected class by predicted class for labeled frames.

No WAV files are required in the repository. Put local recordings under `testdata/real_audio` when measuring changes.

## Training feature export

Export a compact training CSV from the same labeled dataset and analyzer path:

```sh
python3 scripts/export_training_csv.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/lipsync_eval/training.csv \
  --evaluated-only
```

The CSV contains `label`, `file`, `time_seconds`, `eval_frame`, `band_00..band_15`, and `feature_00..feature_30`. The 16-band columns match the current spectral classifier/GMM feature space; the 31-dim `feature_` columns match `FeatureVector.values`.

## GMM training initializer

`train_gmm.py` reads a CSV of finite feature columns and labels, groups rows by class, and exports a small diagonal-GMM JSON without external dependencies:

```sh
python3 scripts/train_gmm.py \
  --input target/lipsync_eval/training.csv \
  --label-column label \
  --feature-set band \
  --mixtures 2 \
  --out target/lipsync_eval/gmm.json
```

`--feature-set band` trains only the `A/I/U/E/O` labels in that fixed order, ignoring non-vowel labels in the CSV. Use it for a trained 16-band GMM comparable to the current prototype classifier. Use the default `feature_` prefix for the 31-dim MFCC/delta/spectral/voicing/f0 vector when training a future 31-dim model.

## Rust GMM export

Export a trained band GMM JSON into Rust constants, rebuild, then evaluate with `--gmm`:

```sh
python3 scripts/export_gmm_rust.py \
  --input target/lipsync_eval/gmm.json \
  --out src/trained_band_gmm.rs
cargo build --release
python3 scripts/evaluate_dataset.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/lipsync_eval_gmm \
  --gmm
```

`LIPSYNC_FLAG_GMM` selects the trained 16-band spectral GMM path; falls back to placeholder if generated arrays are empty. It consumes `normalized_bands[16]` and does not consume the 31-dim `FeatureVector.values`. Accuracy claims require `gmm_model_kind == 2` in `frames.csv`; `gmm_model_kind == 1` is only the buildability fallback before a generated model exists.

## Evaluation comparison

Compare default and GMM evaluation directories produced from the same dataset:

```sh
python3 scripts/compare_evaluations.py \
  --baseline target/eval_default_test \
  --candidate target/eval_gmm_test \
  --out target/eval_compare
```

The report includes `compare_summary.json`, `confusion_delta.csv`, `per_label_metrics.csv`, and `error_breakdown.csv` for raw-vowel, final-class, absorption, pair-confusion, class-switch, and available `gmm_model_kind` deltas/context.

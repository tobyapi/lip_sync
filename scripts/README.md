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

This is a lightweight initializer/export path, not an accuracy guarantee. It needs real labeled feature data exported from the SDK feature extractor before replacing the placeholder model. Use `--feature-set band` for a trained 16-band GMM comparable to the current prototype classifier. Use the default `feature_` prefix for the 31-dim MFCC/delta/spectral/voicing/f0 vector when training a future 31-dim model.

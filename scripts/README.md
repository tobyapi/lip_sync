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

- `frames.csv`: per-frame posterior, best class, jaw opening, confidence, and entropy.
- `summary.json`: top-1/top-2 vowel accuracy, rest rejection, fricative/closed detection when labels exist, average jaw opening, class switches per second, and posterior entropy.
- `confusion_matrix.csv`: expected class by predicted class for labeled frames.

No WAV files are required in the repository. Put local recordings under `testdata/real_audio` when measuring changes.

## GMM training initializer

`train_gmm.py` reads a CSV of finite feature columns and labels, groups rows by class, and exports a small diagonal-GMM JSON without external dependencies:

```sh
python3 scripts/train_gmm.py \
  --input target/lipsync_eval/features.csv \
  --label-column label \
  --feature-prefix feature_ \
  --mixtures 2 \
  --out target/lipsync_eval/gmm.json
```

This is a lightweight initializer/export path, not an accuracy guarantee. It needs real labeled feature data exported from the SDK feature extractor before replacing the placeholder model. Feature rows should include MFCC 1..12, delta MFCC, spectral stats, voicing, and f0 columns with the same order used by `FeatureVector.values`.

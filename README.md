# Lip Sync Analysis SDK

A profile-free, low-latency Rust lip-sync core with a C ABI and Unity binding sources. The SDK analyzes normalized PCM frames and returns a 9-class posterior plus `jaw_open` without requiring per-user MFCC/profile recording.

Implemented scope in this repository:

- Profile-free spectral-envelope vowel evidence.
- Low-latency ring-buffer analysis with time-based windows and arbitrary input chunk sizes.
- C ABI suitable for native plugins and engine bindings.
- Unity binding source under `bindings/unity`.
- Singing mode temporal behavior for sustained notes, slower vowel-to-vowel transitions, and smoother jaw movement.
- Loud voice / shouting / compressed-audio robustness through RMS normalization, soft limiting, clipping/crest detection, adaptive loudness tracking, and compressed-voice posterior priors.
- Optional tiny NN blend over normalized spectral features.
- MFCC/voicing feature extractor for future trained-model experiments.
- Robust rolling loudness normalization for profile-free adaptation, with CMVN helper code reserved for future trained model paths.
- TTS viseme metadata and lyric timing fusion through timed class cues.
- Engine-agnostic mapped output for generic, VRM, ARKit-style, and MetaHuman-style mouth controls.
- Interleaved PCM processing helpers for bindings that receive engine-native multi-channel audio buffers.

Not implemented here:

- Unreal integration package/wrapper. The native mapper API is engine-agnostic and can be consumed by Unreal or other hosts.

## Rust Build

```sh
cargo test
cargo build --release
```

The Windows DLL is emitted at:

```text
target/release/lip_sync.dll
```

For Unity usage, copy the DLL to:

```text
Assets/Plugins/x86_64/lip_sync.dll
```

## Streaming Windows

`LipSyncAnalyzer::process` and the C ABI accept arbitrary PCM chunk sizes. Samples are appended to an internal ring buffer and analyzed with time-based windows: normal mode uses a 25 ms window and 10 ms hop, while singing mode uses a 40 ms window and 10 ms hop. If not enough samples have arrived yet, the analyzer returns the latest safe frame, initially REST. The internal FFT can still use `FRAME_SIZE`, but the input chunk no longer needs to be 1024 samples.
## Output Classes

`LipSyncFrame.posterior` contains probabilities in this order:

```text
REST, CLOSED, A, I, U, E, O, FRICATIVE, OTHER
```

The frame also includes:

- `jaw_open`: normalized mouth opening estimate.
- `vowel_confidence`: confidence for the A/I/U/E/O evidence.
- `f1_hz`, `f2_hz`: debug/visualization formant estimates only.

For engine integration, `LipSyncMappedFrame` provides stable mouth-control weights derived from the 9-class posterior:

- `aa`, `ih`, `ou`, `ee`, `oh`: VRM/Japanese vowel-style viseme weights.
- `mouth_close`, `mouth_funnel`, `mouth_pucker`, `mouth_wide`, `mouth_*`: ARKit-style and MetaHuman-style curve weights.
- `best_class`: native helper output so bindings do not need to duplicate class selection.

Mapper kinds are `0` generic, `1` VRM, `2` ARKit-style, and `3` MetaHuman-style. New integrations should consume the 9-class posterior or mapped output directly.

## Options

```c
typedef struct {
    uint32_t sample_rate;
    uint32_t flags;
    float metadata_weight;
    float smoothing;
    float loudness_adaptation;
} LipSyncOptions;
```

Flags:

```text
1 << 0  Singing mode / slower temporal vowel transitions
1 << 1  Optional tiny NN
1 << 2  Timed cues / metadata fusion
1 << 3  Robust loudness handling
1 << 4  Placeholder 16-band spectral GMM infrastructure path, default off
```

## C ABI

Stateful SDK API:

```c
LipSyncOptions lipsync_default_options(uint32_t sample_rate);
LipSyncOptions lipsync_singing_options(uint32_t sample_rate);
LipSyncAnalyzer* lipsync_create(uint32_t sample_rate, bool singing_mode);
LipSyncAnalyzer* lipsync_create_with_options(LipSyncOptions options);
bool lipsync_process(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, LipSyncFrame* result);
bool lipsync_process_debug(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, LipSyncDebugFrame* result);
bool lipsync_process_at_time(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, float time_seconds, LipSyncFrame* result);
bool lipsync_process_at_time_debug(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, float time_seconds, LipSyncDebugFrame* result);
bool lipsync_process_mapped(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, uint32_t mapper_kind, LipSyncMappedFrame* result);
bool lipsync_process_at_time_mapped(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, float time_seconds, uint32_t mapper_kind, LipSyncMappedFrame* result);
bool lipsync_process_interleaved(LipSyncAnalyzer* analyzer, const float* pcm, size_t frame_count, uint32_t channels, LipSyncFrame* result);
bool lipsync_process_interleaved_at_time(LipSyncAnalyzer* analyzer, const float* pcm, size_t frame_count, uint32_t channels, float time_seconds, LipSyncFrame* result);
bool lipsync_process_interleaved_mapped(LipSyncAnalyzer* analyzer, const float* pcm, size_t frame_count, uint32_t channels, uint32_t mapper_kind, LipSyncMappedFrame* result);
bool lipsync_process_interleaved_at_time_mapped(LipSyncAnalyzer* analyzer, const float* pcm, size_t frame_count, uint32_t channels, float time_seconds, uint32_t mapper_kind, LipSyncMappedFrame* result);
bool lipsync_frame_best_class(const LipSyncFrame* frame, uint32_t* result);
bool lipsync_frame_class_score(const LipSyncFrame* frame, uint32_t class_index, float* result);
bool lipsync_map_frame(const LipSyncFrame* frame, uint32_t mapper_kind, LipSyncMappedFrame* result);
bool lipsync_set_timed_cues(LipSyncAnalyzer* analyzer, const LipSyncTimedCue* cues, size_t len);
bool lipsync_clear_timed_cues(LipSyncAnalyzer* analyzer);
void lipsync_destroy(LipSyncAnalyzer* analyzer);
```

Timed cues use `class_index` values from the 9-class order. Cue kind `1` is TTS viseme metadata and kind `2` is lyric timing. Metadata is blended with audio posterior using `metadata_weight`; lyric cues are slightly softer than TTS viseme cues.

## Unity Usage

See `bindings/unity` for the C# wrapper and microphone analyzer component. The wrapper imports the native plugin as `lip_sync`, so the Unity plugin DLL should be named `lip_sync.dll` on Windows.

Important Unity options:

- `singingMode`: smoother transitions and jaw movement for sustained vocals.
- `enableTinyNn`: blends the optional tiny NN with prototype matching.
- `robustLoudness`: enables shout/compressor-oriented normalization and adaptive gates.
- `enableGmm`: default off; enables the trained 16-band spectral GMM path, falling back to the placeholder model if no generated arrays are present.
- `metadataWeight`: controls TTS/lyric timed cue influence.
- `useAudioSourceTimeForTimedCues`: uses `AudioSource.time` for metadata lookup.

## Evaluation Workflow

Use the real-audio evaluator before claiming accuracy changes. The repository defines a local dataset layout under `testdata/real_audio`, but real WAV files are not required or committed.

```sh
cargo build --release
python3 scripts/evaluate_dataset.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/lipsync_eval
```

The evaluator reads mono or stereo PCM WAV files, downmixes to mono, runs the debug C ABI with configurable chunks, and writes `frames.csv`, `summary.json`, and `confusion_matrix.csv`. The per-frame CSV includes final posterior columns `p_rest` through `p_other`, raw vowel scores `vowel_scores_a` through `vowel_scores_o`, `band_00` through `band_15`, `feature_00` through `feature_30`, feature-space ids, `vowel_confidence`, `activity`, `rms`, `high_ratio`, `zcr`, `flatness`, `compression_likelihood`, and `raw_best_vowel`. The summary includes vowel top-1/top-2 accuracy, rest rejection, fricative and closed detection when labels exist, average jaw opening, class switches per second, and mean posterior entropy. The optional `--gmm` flag evaluates the trained 16-band vowel GMM after `src/trained_band_gmm.rs` has been generated. If the generated arrays are empty, the SDK falls back to the clearly marked placeholder model, which is not an accuracy model.

## Training a 16-Band GMM

Export training rows from a labeled real-audio dataset, train a five-vowel band model in fixed `A/I/U/E/O` order, generate Rust constants, then rebuild and evaluate with `--gmm`:

```sh
cargo build --release
python3 scripts/export_training_csv.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/lipsync_eval/training.csv \
  --evaluated-only
python3 scripts/train_gmm.py \
  --input target/lipsync_eval/training.csv \
  --feature-set band \
  --mixtures 2 \
  --out target/lipsync_eval/band_gmm.json
python3 scripts/export_gmm_rust.py \
  --input target/lipsync_eval/band_gmm.json \
  --out src/trained_band_gmm.rs
cargo build --release
python3 scripts/evaluate_dataset.py \
  --library target/release/liblip_sync.so \
  --dataset testdata/real_audio \
  --out target/eval_band_gmm \
  --gmm
```

`--feature-set band` trains only labels `A`, `I`, `U`, `E`, and `O` in the fixed vowel order used by the runtime. The generated GMM consumes `normalized_bands[16]`; it does not consume the 31-dimensional `FeatureVector.values` path.

## Feature Extractor

The crate includes a `FeatureExtractor` and training CSV export for trained-model experiments. The debug C ABI exposes the current 16-band classifier space and the 31-dimensional feature vector separately. `scripts/export_training_csv.py` writes `label`, `file`, `time_seconds`, `eval_frame`, `band_*`, and `feature_*` columns; `scripts/train_gmm.py --feature-set band` trains from the 16-band space, and `scripts/export_gmm_rust.py` exports that model into `src/trained_band_gmm.rs`. The default `feature_` prefix remains for the 31-dimensional vector. The feature extractor produces:

- 24 mel-like log band energies internally.
- MFCC 1..12, excluding MFCC 0 from vowel-shape classification.
- Delta MFCC from the previous analyzer frame.
- Spectral centroid and spectral tilt.
- High-frequency ratio, zero crossing rate, and spectral flatness.
- Simple voiced confidence.
- Simple autocorrelation f0 estimate.
- `rms_db` for loudness and jaw behavior.

The trained band GMM path intentionally does not consume `FeatureVector.values`: those are 31-dimensional MFCC/voicing features, while the vowel GMM consumes `normalized_bands[16]`. Until a generated trained band model is committed, GMM mode falls back to the placeholder 16-band prototype GMM. The fixed tiny NN also uses the current 16-band shape today, while the feature extractor remains the intended input surface for future trained 31-dimensional model export.

## Closed Detection

CLOSED uses a dedicated low-latency heuristic detector instead of treating every low-confidence vowel as closed. The detector uses short energy valleys, low high-frequency ratio, compact spectral shape, low jaw-openness target, and nearby onset evidence. The default mode is causal and conservative. An internal quality/lookahead mode can boost CLOSED when a 20-30 ms style valley-plus-following-onset pattern is available, but perfect p/b/m detection is impossible from audio-only causal frames.

## Temporal State

`LipSyncAnalyzer` applies a lightweight temporal state machine after posterior scoring. It tracks the current class, hold time, previous time step, and switch confidence. Class-specific minimum holds and hysteresis reduce flicker while still allowing quick CLOSED and FRICATIVE attacks. Singing mode increases vowel-to-vowel hold time and switch margin, so sustained vocals move more slowly than normal speech.

## Adaptive Normalization

The active analyzer uses a rolling loudness tracker to estimate a noise floor, speech-high level, and `normalized_level_01` for jaw/loudness behavior. It uses robust EMA-style low/high tracking rather than storing user profiles. `RollingCmvn` remains available for future trained 31-dimensional model paths, but it is not applied to the 16-band GMM path.

## Classifier Notes

The default vowel evidence path is multi-prototype normalized spectral matching over 16 log-energy bands, optionally blended with a tiny NN. `LIPSYNC_FLAG_GMM` loads `trained_band_vowel_gmm()`, which consumes the same 16-band spectral feature space. If no generated trained arrays are present, it falls back to the hand-written placeholder GMM; that fallback is not an accuracy-improvement mode and should not be used for accuracy claims. Each vowel keeps the original hand-written prototype as a base and adds deterministic pitch, loudness, singing, and microphone-response variants. The classifier intentionally does not use F1/F2 polygon mapping. LPC/formants remain only as debug and auxiliary evidence. Compressed, clipped, and shouted voices use feature smoothing plus a weak broad vowel prior capped at 0.18 so mouth shape is preserved; compression mainly dampens confidence and stabilizes jaw opening instead of forcing an A-heavy distribution.

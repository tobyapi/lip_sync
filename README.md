# Lip Sync Analysis SDK

A profile-free, low-latency Rust lip-sync core with a C ABI and Unity binding sources. The SDK analyzes normalized PCM frames and returns a 9-class posterior plus `jaw_open` without requiring per-user MFCC/profile recording.

Implemented scope in this repository:

- Profile-free spectral-envelope vowel evidence.
- Low-latency 1024-sample frame analysis.
- C ABI suitable for native plugins and engine bindings.
- Unity binding source under `bindings/unity`.
- Singing mode temporal behavior for sustained notes, slower vowel-to-vowel transitions, and smoother jaw movement.
- Loud voice / shouting / compressed-audio robustness through RMS normalization, soft limiting, clipping/crest detection, adaptive loudness tracking, and compressed-voice posterior priors.
- Optional tiny NN blend over normalized spectral features.
- MFCC/voicing feature extractor for GMM and tiny-classifier experiments.
- Rolling CMVN and robust loudness normalization for profile-free adaptation.
- TTS viseme metadata and lyric timing fusion through timed class cues.
- Legacy single-vowel C ABI for compatibility.

Not implemented here:

- Unreal integration.
- VRM mapper.
- ARKit mapper.
- MetaHuman mapper.

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

## Output Classes

`LipSyncFrame.posterior` contains probabilities in this order:

```text
REST, CLOSED, A, I, U, E, O, FRICATIVE, OTHER
```

The frame also includes:

- `jaw_open`: normalized mouth opening estimate.
- `vowel_confidence`: confidence for the A/I/U/E/O evidence.
- `f1_hz`, `f2_hz`: debug/visualization formant estimates only.

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
1 << 4  Diagonal GMM classifier path
```

## C ABI

Legacy compatibility:

```c
bool recognize_vowel(const float* pcm, size_t len, uint32_t sample_rate, Vowel* result);
```

Stateful SDK API:

```c
LipSyncAnalyzer* lipsync_create(uint32_t sample_rate, bool singing_mode);
LipSyncAnalyzer* lipsync_create_with_options(LipSyncOptions options);
bool lipsync_process(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, LipSyncFrame* result);
bool lipsync_process_at_time(LipSyncAnalyzer* analyzer, const float* pcm, size_t len, float time_seconds, LipSyncFrame* result);
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

The evaluator reads mono or stereo PCM WAV files, downmixes to mono, runs the C ABI with configurable chunks, and writes `frames.csv`, `summary.json`, and `confusion_matrix.csv`. The summary includes vowel top-1/top-2 accuracy, rest rejection, fricative and closed detection when labels exist, average jaw opening, class switches per second, and mean posterior entropy.

## Feature Extractor

The stateful analyzer owns a `FeatureExtractor` for richer classifier paths. It produces:

- 24 mel-like log band energies internally.
- MFCC 1..12, excluding MFCC 0 from vowel-shape classification.
- Delta MFCC from the previous analyzer frame.
- Spectral centroid and spectral tilt.
- High-frequency ratio, zero crossing rate, and spectral flatness.
- Simple voiced confidence.
- Simple autocorrelation f0 estimate.
- `rms_db` for loudness and jaw behavior.

GMM mode consumes the richer `FeatureVector.values`. The fixed tiny NN still uses the legacy 16-band shape today, but the feature extractor is the intended input surface for future tiny-model training/export.
## Closed Detection

CLOSED uses a dedicated low-latency heuristic detector instead of treating every low-confidence vowel as closed. The detector uses short energy valleys, low high-frequency ratio, compact spectral shape, low jaw-openness target, and nearby onset evidence. The default mode is causal and conservative. An internal quality/lookahead mode can boost CLOSED when a 20-30 ms style valley-plus-following-onset pattern is available, but perfect p/b/m detection is impossible from audio-only causal frames.
## Temporal State

`LipSyncAnalyzer` applies a lightweight temporal state machine after posterior scoring. It tracks the current class, hold time, previous time step, and switch confidence. Class-specific minimum holds and hysteresis reduce flicker while still allowing quick CLOSED and FRICATIVE attacks. Singing mode increases vowel-to-vowel hold time and switch margin, so sustained vocals move more slowly than normal speech.
## Adaptive Normalization

GMM mode applies rolling cepstral mean/variance normalization (CMVN) to `FeatureVector.values`. CMVN updates only on reliable voiced frames: not REST, not FRICATIVE, finite feature values, and not strongly clipped/compressed. This is adaptive profile-free normalization for microphone/EQ/recording drift; it is not user MFCC profile recording.

A rolling loudness tracker estimates a noise floor, speech-high level, and `normalized_level_01` for jaw/loudness behavior. It uses robust EMA-style low/high tracking rather than storing user profiles.
## Classifier Notes

The default vowel evidence path is multi-prototype normalized spectral matching over 16 log-energy bands, optionally blended with a tiny NN. A diagonal GMM infrastructure path is available with `LIPSYNC_FLAG_GMM`, currently seeded from the same hand-written prototype family until trained data is available. Each vowel keeps the original hand-written prototype as a base and adds deterministic pitch, loudness, singing, and microphone-response variants. The placeholder GMM is infrastructure only; real accuracy claims should come from the evaluation workflow and a trained exported model. The classifier intentionally does not use F1/F2 polygon mapping. LPC/formants remain only as debug and auxiliary evidence. Compressed, clipped, and shouted voices use feature smoothing plus a weak broad vowel prior capped at 0.18 so mouth shape is preserved; compression mainly dampens confidence and stabilizes jaw opening instead of forcing an A-heavy distribution.

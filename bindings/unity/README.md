# Unity Binding

This folder contains lightweight Unity C# bindings for the Rust `lip_sync` C ABI.

## Setup

1. Build the native library:

```sh
cargo build --release
```

2. Copy the native library into Unity:

```text
Assets/Plugins/x86_64/lip_sync.dll
```

3. Copy `LipSync.cs` and `MicrophoneLipSyncAnalyzer.cs` into a Unity `Assets` folder.

4. Add `MicrophoneLipSyncAnalyzer` to a GameObject with an `AudioSource` that is playing microphone, TTS, song, or voice audio.

## Timed Metadata

Use `SetTimedCues` with `LipSyncTimedCue` entries for TTS viseme metadata or lyric timing. The cue `classIndex` uses the same 9-class order as the Rust SDK: `REST`, `CLOSED`, `A`, `I`, `U`, `E`, `O`, `FRICATIVE`, `OTHER`.

## Classifier Flags

The C# binding exposes `LipSyncOptionsFlags.Gmm` and the microphone component has an `enableGmm` toggle. This selects the placeholder diagonal-GMM classifier path in the native SDK; it is intended for trained model experiments and does not imply better accuracy until measured with real labeled WAV data.

## Audio Thread Usage

`LipSyncAnalyzer.Process` can accept arbitrary PCM callback chunk sizes. For lowest Unity latency, copy PCM samples from `OnAudioFilterRead` into a lock-free or otherwise thread-safe buffer, pass chunks to the native analyzer, and consume the latest `LipSyncFrame` on the main thread. Do not call Unity APIs such as `AudioSource`, `Debug.Log`, or scene object methods from the audio thread.

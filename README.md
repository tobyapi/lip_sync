# Lip Sync Analysis Library

A Rust library for performing lip-sync analysis on audio data, with a focus on vowel recognition.

## Features

- **Vowel Recognition**: Identifies vowels (A, I, U, E, O) from raw PCM audio data.
- **LPC Analysis**: Implements Linear Predictive Coding (LPC) analysis, including autocorrelation, Levinson-Durbin algorithm, and conversion to cepstrum.
- **Formant Analysis**: Detects formants from the LPC spectrum.
- **Advanced Vowel Mapping**: Uses complex, non-convex polygons for robust F1/F2 formant mapping to vowels.
- **Noise Reduction**: Includes multiple strategies for noise reduction:
    - Voiced/Unvoiced frame detection.
    - Experimental zero-phase (cepstral) filtering.
- **Cross-Platform**: Written in Rust, can be compiled for multiple platforms.
- **C FFI**: Exposes a C-compatible foreign function interface (FFI) for easy integration with other languages.

## Building

This project is a Rust library that compiles to a dynamic library (`.dll`, `.so`, `.dylib`).

### Windows (Cross-compilation from Linux)

To build the Windows DLL from a Debian-based Linux environment, you need the MinGW toolchain.

1.  **Install Toolchain:**
    ```sh
    sudo apt-get update
    sudo apt-get install gcc-mingw-w64-x86-64
    ```

2.  **Add Rust Target:**
    ```sh
    rustup target add x86_64-pc-windows-gnu
    ```

3.  **Build:**
    ```sh
    cargo build --release --target x86_64-pc-windows-gnu
    ```
    The final DLL will be located at `target/x86_64-pc-windows-gnu/release/lip_sync.dll`.

## Usage (C FFI)

The library exposes the following C-compatible functions.

### `recognize_vowel`

Analyzes a chunk of PCM audio data and attempts to recognize the vowel being spoken.

**Signature:**
```c
bool recognize_vowel(const float* pcm, size_t len, uint32_t sample_rate, Vowel* result);
```

-   `pcm`: A pointer to an array of `float` audio samples, normalized between -1.0 and 1.0.
-   `len`: The number of samples in the `pcm` array.
-   `sample_rate`: The sample rate of the audio (e.g., 44100).
-   `result`: A pointer to a `Vowel` enum where the result will be stored.
-   **Returns:** `true` if a vowel was successfully recognized, `false` otherwise (e.g., if the frame was determined to be unvoiced or no vowel could be classified).

**`Vowel` Enum:**
```c
typedef enum {
    A = 0,
    I = 1,
    U = 2,
    E = 3,
    O = 4,
} Vowel;
```

## Usage (Unity C#)

Here is an example of how to call the `recognize_vowel` function from C# in Unity.

1.  Build the `lip_sync.dll` for your target platform (e.g., Windows x86_64) and place it in the `Assets/Plugins` folder of your Unity project.
2.  Use the following C# script to interface with the library.

```csharp
using System;
using System.Runtime.InteropServices;
using UnityEngine;

public enum Vowel
{
    A,
    I,
    U,
    E,
    O
}

public class LipSync
{
    public static bool TryRecognizeVowel(float[] pcm, int sampleRate, out Vowel vowel)
    {
        return Bindings.recognize_vowel(pcm, (UIntPtr)pcm.Length, (uint)sampleRate, out vowel);
    }

    static class Bindings
    {
        [DllImport("lip_sync")]
        internal static extern bool recognize_vowel(float[] pcm_data, UIntPtr len, uint sample_rate, out Vowel result);
    }
}
```

## Modules

-   `lpc`: Core Linear Predictive Coding (LPC) functions like `autocorrelate`, `levinson_durbin`, etc.
-   `vowel`: Contains the main vowel recognition logic, including formant detection, voiced/unvoiced detection, and F1/F2 mapping.
-   `geometry`: Helper functions for computational geometry, such as point-in-polygon tests for formant mapping.
-   `zero_phase`: Experimental noise reduction based on cepstral processing.
-   `lattice`: Implementation of LPC lattice filters for analysis and synthesis (currently not exposed in the public API).
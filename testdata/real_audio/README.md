# Real Audio Test Dataset

This directory defines the local dataset layout used by `scripts/evaluate_dataset.py`. Real WAV clips are intentionally not required in the repository.

Recommended layout:

```text
testdata/real_audio/
  vowels/
    speaker01/
      normal/a_01.wav
      normal/i_01.wav
      normal/u_01.wav
      normal/e_01.wav
      normal/o_01.wav
      loud/a_01.wav
      singing/a_01.wav
  non_vowels/
    silence/
    breath/
    fricative/
    closed/
    noise/
```

## Recording Guidance

- Record mono or stereo PCM WAV files. The evaluator downmixes stereo to mono.
- Prefer 16 kHz, 24 kHz, 44.1 kHz, or 48 kHz sample rates.
- Use sustained, stable vowels for `vowels/*`; avoid evaluating the attack and release portions.
- Include normal speech, loud voice, compressed/limited voice, and singing clips when possible.
- Keep clips short and consistently trimmed. One to three seconds is enough for many checks.
- Do not commit private or licensed voice data unless you are certain it can be redistributed.

## Labels

Vowel labels are inferred from the filename prefix:

```text
a_01.wav -> A
i_01.wav -> I
u_01.wav -> U
e_01.wav -> E
o_01.wav -> O
```

Non-vowel labels can be inferred from either filename prefixes or directory names:

```text
silence -> REST
rest -> REST
closed -> CLOSED
fricative -> FRICATIVE
breath -> OTHER
noise -> OTHER
other -> OTHER
```

For sustained vowel clips, the evaluator only scores the stable center region by default. Use `--center-region` to change the fraction of the clip that is evaluated.

pub mod lpc;
pub mod vowel;

use crate::vowel::{recognize_vowel_from_pcm, Vowel};
use libc::size_t;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn hoge() -> i32 {
    100
}

#[unsafe(no_mangle)]
pub extern "C" fn hoge2(pcm_data: *const f32, len: size_t) -> f32 {
    let numbers = unsafe {
        assert!(!pcm_data.is_null());
        slice::from_raw_parts(pcm_data, len as usize)
    };

    numbers
        .iter()
        .sum()
}

#[unsafe(no_mangle)]
pub extern "C" fn hoge3(out: *mut Vowel) -> bool {
    let result = Some(Vowel::U);
    if let Some(vowel) = result {
        unsafe {
            *out = vowel;
        }
        true
    } else {
        false
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn recognize_vowel(pcm: *const f32, len: size_t, sample_rate: u32, result: *mut Vowel) -> bool {
    let pcm_data = unsafe {
        slice::from_raw_parts(pcm, len as usize)
    };
    if let Some(vowel) = recognize_vowel_from_pcm(pcm_data, sample_rate) {
        unsafe { *result = vowel; }
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use hound::WavReader;

    #[test]
    fn test_vowel_recognition() {
        let mut reader = WavReader::open("testdata/test.wav").unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let sample_rate = reader.spec().sample_rate;
        println!("Sample rate: {}", sample_rate);

        let order = 24;
        let chunk_size = 512;
        let fft_size = 1024;

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            if chunk.len() < chunk_size {
                continue;
            }
            println!("--- Chunk {} ---", i);
            let mut signal_chunk = chunk.to_vec();
            crate::lpc::hamming_window(&mut signal_chunk);

            let mut acf = crate::lpc::autocorrelate(&signal_chunk);
            let acf0 = acf[0];
            if acf0 > 0.0 {
                for val in &mut acf {
                    *val /= acf0;
                }
            }

            if let Some((alpha, err)) = crate::lpc::levinson_durbin(&acf, order) {
                let gain = err * acf0;
                
                let spectral_envelope = crate::lpc::lpc_to_spectral_envelope(&alpha, gain, fft_size);
                let formants = crate::vowel::find_formants(&spectral_envelope, sample_rate, 5);

                if !formants.is_empty() {
                    println!("Detected formants:");
                    for f in &formants {
                        println!("  {:.1} Hz (Magnitude: {:.2})", f.frequency, f.magnitude);
                    }

                    if let Some(vowel) = crate::vowel::recognize_vowel(&formants) {
                        println!("Recognized vowel: {:?}", vowel);
                    } else {
                        println!("Could not recognize vowel.");
                    }
                } else {
                    println!("No formants detected for this chunk.");
                }
            } else {
                println!("Levinson-Durbin algorithm failed for this chunk.");
            }
        }
    }

    #[test]
    fn test_recognize_vowel_from_pcm() {
        // 1. Read audio file
        let mut reader = WavReader::open("testdata/test.wav").unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let sample_rate = reader.spec().sample_rate;

        // 2. Select a chunk of audio data
        // Find the chunk with the highest energy to get a stable result
        //let chunk_size = 1024;
        let chunk_size = 4096;
        let num_chunks = samples.len() / chunk_size;
        let mut best_chunk_start = 0;
        let mut max_energy = 0.0;
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;
            let energy = samples[start..end].iter().map(|&s| s * s).sum();
            if energy > max_energy {
                max_energy = energy;
                best_chunk_start = start;
            }
        }
        let pcm_chunk = &samples[best_chunk_start..best_chunk_start + chunk_size];

        // 3. Call the new function to recognize the vowel
        let vowel = crate::vowel::recognize_vowel_from_pcm(pcm_chunk, sample_rate);

        // 4. Print the result
        println!("--- Test recognize_vowel_from_pcm ---");
        println!("Sample rate: {}", sample_rate);
        println!("Chunk size: {}", pcm_chunk.len());
        if let Some(ref v) = vowel {
            println!("Recognized vowel: {:?}", v);
        } else {
            println!("Could not recognize vowel.");
        }
        // Add an assertion to make it a real test
        assert!(vowel.is_some());
    }
}
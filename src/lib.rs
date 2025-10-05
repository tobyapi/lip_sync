pub mod geometry;
pub mod lpc;
pub mod vowel;

use crate::vowel::{recognize_vowel_from_pcm, Vowel};
use libc::size_t;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn recognize_vowel(
    pcm: *const f32,
    len: size_t,
    sample_rate: u32,
    result: *mut Vowel,
) -> bool {
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    if let Some(vowel) = recognize_vowel_from_pcm(pcm_data, sample_rate) {
        unsafe {
            *result = vowel;
        }
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

        let chunk_size = 512;

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            if chunk.len() < chunk_size {
                continue;
            }
            println!("--- Chunk {} ---", i);

            match crate::vowel::recognize_vowel_from_pcm(chunk, sample_rate) {
                Some(vowel) => println!("Recognized vowel: {:?}", vowel),
                None => println!("Could not recognize vowel."),
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
        let chunk_size = 1024;
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
use crate::geometry::{is_inside_polygon, Point};
use crate::lpc;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Vowel {
    A,
    I,
    U,
    E,
    O,
}

#[derive(Debug)]
pub struct Formant {
    pub frequency: f32,
    pub magnitude: f32,
}

/// Detects formants from a spectral envelope.
///
/// It identifies formant frequencies by finding peaks (local maxima) in the spectrum.
///
/// # Arguments
/// * `spectrum` - A vector of the spectral envelope.
/// * `sample_rate` - The sample rate of the original audio.
/// * `max_formants` - The maximum number of formants to detect.
///
/// # Returns
/// * `Vec<Formant>` - A vector of detected formants, containing frequency and magnitude information.
pub fn find_formants(spectrum: &[f32], sample_rate: u32, max_formants: usize) -> Vec<Formant> {
    let mut formants = Vec::new();
    let fft_size = spectrum.len();

    // Limit search to 5000 Hz
    let max_freq_index = (5000.0 / (sample_rate as f32 / fft_size as f32)) as usize;
    let search_range = 1..max_freq_index.min(fft_size / 2 - 1);

    if search_range.is_empty() {
        return formants;
    }

    for i in search_range {
        // Simple peak detection
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            let freq = i as f32 * sample_rate as f32 / fft_size as f32;
            formants.push(Formant {
                frequency: freq,
                magnitude: spectrum[i],
            });
        }
    }

    // Peaks are detected in frequency order, so we don't sort by magnitude.
    // This preserves the natural order of F1, F2, ...
    formants.truncate(max_formants);

    formants
}

/// Determines the vowel from the first and second formants.
///
/// It calculates the Euclidean distance between the detected formant frequencies and the center values
/// of the F1/F2 frequencies for each predefined Japanese vowel, and identifies the closest one.
///
/// # Arguments
/// * `formants` - A vector of detected formants.
///
/// # Returns
/// * `Option<Vowel>` - Returns the closest vowel if two or more formants are detected, otherwise `None`.
pub fn find_vowel(formants: &[Formant]) -> Option<Vowel> {
    if formants.len() < 2 {
        return None;
    }

    let formant_point = Point { x: formants[0].frequency, y: formants[1].frequency };

    const VOWEL_A: &[Point] = &[
        Point { x: 500.0, y: 0.0 },
        Point { x: 1500.0, y: 0.0 },
        Point { x: 1500.0, y: 2800.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 800.0, y: 2800.0 },
        Point { x: 580.0, y: 1900.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 600.0, y: 1500.0 },
    ];
    const VOWEL_I: &[Point] = &[
        Point { x: 0.0, y: 2000.0 },
        Point { x: 200.0, y: 2100.0 },
        Point { x: 300.0, y: 3000.0 },
        Point { x: 800.0, y: 2800.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 500.0, y: 3200.0 },
        Point { x: 550.0, y: 5000.0 },
        Point { x: 0.0, y: 5000.0 },
    ];
    const VOWEL_U: &[Point] = &[
        Point { x: 0.0, y: 800.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 580.0, y: 1900.0 },
        Point { x: 200.0, y: 2100.0 },
        Point { x: 0.0, y: 2000.0 },
    ];
    const VOWEL_E: &[Point] = &[
        Point { x: 200.0, y: 2100.0 },
        Point { x: 580.0, y: 1900.0 },
        Point { x: 800.0, y: 2800.0 },
        Point { x: 300.0, y: 3000.0 },
    ];
    const VOWEL_O_1: &[Point] = &[
        Point { x: 0.0, y: 0.0 },
        Point { x: 500.0, y: 0.0 },
        Point { x: 650.0, y: 1500.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 0.0, y: 800.0 },
    ];
    const VOWEL_O_2: &[Point] = &[
        Point { x: 500.0, y: 3200.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 1500.0, y: 2800.0 },
        Point { x: 1500.0, y: 5000.0 },
        Point { x: 550.0, y: 5000.0 },
    ];

    let vowel_polygons: &[(&[Point], Vowel)] = &[
        (VOWEL_A, Vowel::A),
        (VOWEL_I, Vowel::I),
        (VOWEL_U, Vowel::U),
        (VOWEL_E, Vowel::E),
        (VOWEL_O_1, Vowel::O),
        (VOWEL_O_2, Vowel::O),
    ];

    for (polygon, vowel) in vowel_polygons {
        if is_inside_polygon(formant_point, polygon) {
            return Some(*vowel);
        }
    }

    None
}

/// Applies a pre-emphasis filter to the signal.
fn pre_emphasis(signal: &mut [f32], coefficient: f32) {
    if signal.len() < 2 {
        return;
    }
    // Process from the end to use the original sample values
    for i in (1..signal.len()).rev() {
        signal[i] = signal[i] - coefficient * signal[i - 1];
    }
}

/// Analyzes a PCM audio chunk and recognizes the most likely vowel.
///
/// This function encapsulates the entire process of vowel recognition from a raw audio signal frame.
/// It performs LPC analysis, formant detection, and vowel classification.
///
/// # Arguments
/// * `pcm_data` - A slice of f32 representing the audio chunk. Recommended lengths are 512, 1024, or 2048.
/// * `sample_rate` - The sample rate of the audio data.
///
/// # Returns
/// * `Option<Vowel>` - The recognized vowel, or `None` if recognition fails (e.g., no formants detected).
pub fn recognize_vowel_from_pcm(pcm_data: &[f32], sample_rate: u32) -> Option<Vowel> {
    let order = (sample_rate / 1000) as usize + 4; // Dynamically determine LPC order
    let fft_size = 1024; // FFT size for spectral envelope calculation

    if pcm_data.is_empty() {
        return None;
    }

    let mut signal_chunk = pcm_data.to_vec();
    
    // Apply pre-emphasis filter
    pre_emphasis(&mut signal_chunk, 0.97);
    
    // Apply Hamming window
    lpc::hamming_window(&mut signal_chunk);

    let mut acf = lpc::autocorrelate(&signal_chunk);
    let acf0 = acf[0];
    if acf0 > 0.0 {
        for val in &mut acf {
            *val /= acf0;
        }
    } else {
        // If energy is zero, no signal is present.
        return None;
    }

    if let Some((alpha, err)) = lpc::levinson_durbin(&acf, order) {
        let gain = err * acf0;

        let spectral_envelope = lpc::lpc_to_spectral_envelope(&alpha, gain, fft_size);
        let formants = find_formants(&spectral_envelope, sample_rate, 5);

        find_vowel(&formants)
    } else {
        None
    }
}
use num_complex::Complex;
use rustfft::FftPlanner;

/// Computes the autocorrelation of an audio signal.
///
/// Autocorrelation is a measure of how similar a signal is to a delayed copy of itself as a function of delay.
/// This calculation is required as the first step in LPC (Linear Predictive Coding) analysis.
///
/// # Arguments
/// * `signal` - A slice of the input audio signal. The values should preferably be normalized f32.
///
/// # Returns
/// * `Vec<f32>` - A vector of the calculated autocorrelation values.
pub fn autocorrelate(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut acf = vec![0.0; n];
    for i in 0..n {
        for j in 0..(n - i) {
            acf[i] += signal[j] * signal[j + i];
        }
    }
    acf
}

/// Applies a Hamming window to the signal.
///
/// A window function is used to prevent spectral leakage when cutting out a short frame of a signal.
/// The Hamming window is one of the most commonly used window functions in audio analysis.
///
/// # Arguments
/// * `signal` - A slice of the signal to apply the window to. This function modifies the signal directly.
pub fn hamming_window(signal: &mut [f32]) {
    let n = signal.len();
    for i in 0..n {
        signal[i] *= 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
    }
}

/// Calculates LPC coefficients from autocorrelation values using the Levinson-Durbin algorithm.
///
/// This algorithm can efficiently find the LPC coefficients (alpha) and the prediction error (err).
/// The LPC coefficients indicate how a sample at a certain point in time can be predicted from previous samples.
///
/// # Arguments
/// * `acf` - A vector of autocorrelation values. `acf[0]` corresponds to the signal's energy. It is preferably normalized.
/// * `order` - The order of the LPC analysis. It determines how many past samples are used for prediction.
///
/// # Returns
/// * `Option<(Vec<f32>, f32)>` - If the calculation is successful, it returns a tuple containing a vector of LPC coefficients (alpha) and the final prediction error (err).
///   Returns `None` if the calculation is unstable, such as when the prediction error becomes zero.
pub fn levinson_durbin(acf: &[f32], order: usize) -> Option<(Vec<f32>, f32)> {
    if acf.is_empty() {
        return None;
    }
    let mut alpha = vec![0.0; order + 1];
    let mut err = acf[0];
    if err == 0.0 {
        return None;
    }

    alpha[0] = 1.0;

    for i in 1..=order {
        let mut k = -acf[i];
        for j in 1..i {
            k -= alpha[j] * acf[i - j];
        }
        k /= err;

        let old_alpha = alpha.clone();
        alpha[i] = k;
        for j in 1..i {
            alpha[j] = old_alpha[j] + k * old_alpha[i - j];
        }
        err *= 1.0 - k * k;
        if err == 0.0 {
            return None;
        }
    }

    Some((alpha, err))
}

/// Calculates cepstrum coefficients from LPC coefficients.
///
/// The cepstrum is a feature that represents the spectral envelope of the voice (characteristics of the vocal tract).
/// This function directly converts LPC coefficients to cepstrum using a recursive formula.
///
/// # Arguments
/// * `alpha` - A vector of LPC coefficients obtained from `levinson_durbin`.
/// * `err` - The prediction error gain, also obtained from `levinson_durbin`.
/// * `num_coeffs` - The number of cepstrum coefficients to calculate.
///
/// # Returns
/// * `Vec<f32>` - A vector of the calculated cepstrum coefficients.
pub fn lpc_to_cepstrum(alpha: &[f32], err: f32, num_coeffs: usize) -> Vec<f32> {
    let p = alpha.len() - 1;
    let mut c = vec![0.0; num_coeffs];

    if num_coeffs > 0 {
        if err > 0.0 {
            c[0] = err.ln();
        } else {
            c[0] = f32::NEG_INFINITY;
        }
    }

    let a = &alpha;

    for m in 1..num_coeffs {
        let mut sum = 0.0;
        for k in 1..m {
            if k < a.len() { // Ensure k is within bounds for a
                sum += a[k] * c[m - k];
            }
        }
        if m <= p {
            c[m] = -a[m] - sum;
        } else {
            c[m] = -sum;
        }
    }

    c
}

/// Calculates the spectral envelope from LPC coefficients.
///
/// The spectral envelope is calculated by taking the inverse Fourier transform of the LPC coefficients.
///
/// # Arguments
/// * `alpha` - A vector of LPC coefficients.
/// * `gain` - The prediction error gain.
/// * `fft_size` - The size of the FFT.
///
/// # Returns
/// * `Vec<f32>` - The calculated spectral envelope (log magnitude).
pub fn lpc_to_spectral_envelope(alpha: &[f32], gain: f32, fft_size: usize) -> Vec<f32> {
    let mut a = vec![0.0; fft_size];
    a[0..alpha.len()].copy_from_slice(alpha);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer: Vec<Complex<f32>> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    fft.process(&mut buffer);

    let log_gain = if gain > 0.0 {
        gain.ln()
    } else {
        f32::NEG_INFINITY
    };

    buffer
        .iter()
        .map(|c| log_gain - 2.0 * c.norm().ln())
        .collect()
}

/// Applies liftering to the cepstrum coefficients.
///
/// By keeping only the low-frequency cepstrum coefficients and zeroing out the high-frequency ones,
/// the spectral envelope is smoothed, and source information (like pitch) is removed.
///
/// # Arguments
/// * `cepstrum` - A vector of cepstrum coefficients.
/// * `lifter_length` - The number of low-frequency cepstrum coefficients to keep.
///
/// # Returns
/// * `Vec<f32>` - A vector of the liftered cepstrum coefficients.
pub fn lifter_cepstrum(mut cepstrum: Vec<f32>, lifter_length: usize) -> Vec<f32> {
    for i in lifter_length..cepstrum.len() {
        cepstrum[i] = 0.0;
    }
    cepstrum
}

/// Reconstructs LPC coefficients from liftered cepstrum coefficients.
///
/// # Arguments
/// * `cepstrum` - A vector of liftered cepstrum coefficients (c[0] is the log gain).
/// * `lpc_order` - The order of the LPC coefficients to reconstruct.
///
/// # Returns
/// * `Vec<f32>` - A vector of the reconstructed LPC coefficients (a[0] is 1.0).
pub fn cepstrum_to_lpc(cepstrum: &[f32], lpc_order: usize) -> Vec<f32> {
    let mut a = vec![0.0; lpc_order + 1];
    a[0] = 1.0; // a[0] is always 1.0 for LPC coefficients

    for m in 1..=lpc_order {
        let mut sum = 0.0;
        for k in 1..m {
            sum += a[k] * cepstrum[m - k];
        }
        if m < cepstrum.len() {
            a[m] = -cepstrum[m] - sum;
        } else {
            a[m] = -sum;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;

    #[test]
    fn test_lpc_chain() {
        // This is an integration test for the LPC analysis chain.
        // It's based on the test_cepstrum from lib.rs.

        // --- 1. Read audio file ---
        let mut reader = WavReader::open("testdata/test.wav").unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();

        // --- 2. Select analysis frame ---
        let order = 24;
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

        let mut signal_chunk = samples[best_chunk_start..best_chunk_start + chunk_size].to_vec();

        // --- 3. Pre-process for LPC analysis ---
        hamming_window(&mut signal_chunk);

        // --- 4. Execute cepstrum extraction ---
        let mut acf = autocorrelate(&signal_chunk);

        let acf0 = acf[0];
        if acf0 > 0.0 {
            for val in &mut acf {
                *val /= acf0;
            }
        }

        if let Some((alpha, err)) = levinson_durbin(&acf, order) {
            let gain = err * acf0;
            let cepstrum = lpc_to_cepstrum(&alpha, gain, order + 1);

            // --- 5. Display results ---
            println!("Cepstrum Coefficients (first 5):");
            for (i, &c) in cepstrum.iter().take(5).enumerate() {
                println!("  c[{}]: {}", i, c);
            }
            // Basic assertion to check if cepstrum was calculated
            assert!(!cepstrum.is_empty());
            assert_ne!(cepstrum[0], 0.0);
        } else {
            panic!("Levinson-Durbin algorithm failed");
        }
    }

    #[test]
    fn test_autocorrelate_simple() {
        let signal = [1.0, 2.0, 3.0];
        let acf = autocorrelate(&signal);
        let expected = vec![14.0, 8.0, 3.0];
        assert_eq!(acf.len(), expected.len());
        for (a, b) in acf.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hamming_window_simple() {
        let mut signal = vec![1.0; 3];
        hamming_window(&mut signal);
        let expected = vec![0.08, 1.0, 0.08];
        assert_eq!(signal.len(), expected.len());
        for (a, b) in signal.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lifter_cepstrum() {
        let cepstrum = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let liftered = lifter_cepstrum(cepstrum, 3);
        assert_eq!(liftered, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cepstrum_to_lpc_zeros() {
        let lpc_order = 10;
        let cepstrum = vec![0.0; lpc_order + 1]; // c[0] is log gain, others are 0
        let lpc = cepstrum_to_lpc(&cepstrum, lpc_order);

        let mut expected_lpc = vec![0.0; lpc_order + 1];
        expected_lpc[0] = 1.0;

        assert_eq!(lpc.len(), expected_lpc.len());
        for (a, b) in lpc.iter().zip(expected_lpc.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cepstrum_to_lpc_known_value() {
        // c = [0, -a1, -a2 + a1^2]
        // a = [1, a1, a2]
        let a1 = 0.5;
        let a2 = -0.25;
        let c1 = -a1; // -0.5
        let c2 = -a2 + a1 * a1; // 0.25 + 0.25 = 0.5
        let cepstrum = vec![0.0, c1, c2];
        let lpc_order = 2;
        let lpc = cepstrum_to_lpc(&cepstrum, lpc_order);

        let expected_lpc = vec![1.0, a1, a2];

        assert_eq!(lpc.len(), expected_lpc.len());
        for (a, b) in lpc.iter().zip(expected_lpc.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_cepstrum_lpc_roundtrip() {
        let lpc_order = 12;
        // Generate some plausible random LPC coefficients
        let original_lpc: Vec<f32> = (0..=lpc_order)
            .map(|i| {
                if i == 0 {
                    1.0
                } else {
                    // small random values
                    (rand::random::<f32>() - 0.5) * 0.5
                }
            })
            .collect();

        let err_gain = 1.0; // Assume unity gain for simplicity
        let num_coeffs = lpc_order + 1;

        // LPC -> Cepstrum
        let cepstrum = lpc_to_cepstrum(&original_lpc, err_gain, num_coeffs);

        // Cepstrum -> LPC
        let reconstructed_lpc = cepstrum_to_lpc(&cepstrum, lpc_order);

        // Compare
        assert_eq!(original_lpc.len(), reconstructed_lpc.len());
        for (orig, recon) in original_lpc.iter().zip(reconstructed_lpc.iter()) {
            assert!(
                (orig - recon).abs() < 1e-4,
                "Mismatch: {} vs {}",
                orig,
                recon
            );
        }
    }
}

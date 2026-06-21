use num_complex::Complex;
use rustfft::FftPlanner;

pub const FEATURE_FFT_SIZE: usize = 1024;
pub const NUM_MEL_BANDS: usize = 24;
pub const NUM_MFCC: usize = 12;
pub const FEATURE_VECTOR_LEN: usize = NUM_MFCC * 2 + 7;

const EPSILON: f32 = 1.0e-8;

#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: Vec<f32>,
    pub rms_db: f32,
    pub high_ratio: f32,
    pub zero_crossing_rate: f32,
    pub spectral_flatness: f32,
    pub voiced_confidence: f32,
    pub f0_hz: f32,
}

impl FeatureVector {
    pub fn empty() -> Self {
        Self {
            values: vec![0.0; FEATURE_VECTOR_LEN],
            rms_db: -120.0,
            high_ratio: 0.0,
            zero_crossing_rate: 0.0,
            spectral_flatness: 0.0,
            voiced_confidence: 0.0,
            f0_hz: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FeatureExtractor {
    previous_mfcc: Option<[f32; NUM_MFCC]>,
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extract(&mut self, pcm_data: &[f32], sample_rate: u32) -> FeatureVector {
        let mut vector = extract_features(pcm_data, sample_rate, self.previous_mfcc);
        let mut current_mfcc = [0.0; NUM_MFCC];
        current_mfcc.copy_from_slice(&vector.values[..NUM_MFCC]);
        self.previous_mfcc = Some(current_mfcc);
        sanitize_feature_vector(&mut vector);
        vector
    }
}

pub fn extract_features(
    pcm_data: &[f32],
    sample_rate: u32,
    previous_mfcc: Option<[f32; NUM_MFCC]>,
) -> FeatureVector {
    if pcm_data.is_empty() || sample_rate == 0 {
        return FeatureVector::empty();
    }

    let rms = calculate_rms(pcm_data);
    let rms_db = 20.0 * (rms + EPSILON).log10();
    let zero_crossing_rate = calculate_zero_crossing_rate(pcm_data);
    let mut frame = prepare_frame(pcm_data);
    hamming_window(&mut frame);
    let spectrum = power_spectrum(&frame);
    let bands = mel_like_band_energies(&spectrum, sample_rate, NUM_MEL_BANDS);
    let log_bands = log_energies(&bands);
    let mfcc = mfcc_without_energy(&log_bands);
    let delta = delta_mfcc(mfcc, previous_mfcc);
    let total_energy = bands.iter().sum::<f32>() + EPSILON;
    let high_ratio = bands[(NUM_MEL_BANDS * 2 / 3)..].iter().sum::<f32>() / total_energy;
    let spectral_flatness = spectral_flatness(&bands);
    let spectral_centroid = spectral_centroid(&spectrum, sample_rate);
    let spectral_tilt = spectral_tilt(&log_bands);
    let (f0_hz, f0_confidence) = estimate_f0_autocorrelation(pcm_data, sample_rate);
    let voiced_confidence =
        voiced_confidence(rms, zero_crossing_rate, spectral_flatness, f0_confidence);

    let nyquist = (sample_rate as f32 * 0.5).max(1.0);
    let mut values = Vec::with_capacity(FEATURE_VECTOR_LEN);
    values.extend_from_slice(&mfcc);
    values.extend_from_slice(&delta);
    values.push((spectral_centroid / nyquist).clamp(0.0, 1.0));
    values.push(spectral_tilt.clamp(-4.0, 4.0) * 0.25);
    values.push(high_ratio.clamp(0.0, 1.0));
    values.push(zero_crossing_rate.clamp(0.0, 1.0));
    values.push(spectral_flatness.clamp(0.0, 1.0));
    values.push(voiced_confidence.clamp(0.0, 1.0));
    values.push((f0_hz / 500.0).clamp(0.0, 1.5));

    let mut vector = FeatureVector {
        values,
        rms_db,
        high_ratio,
        zero_crossing_rate,
        spectral_flatness,
        voiced_confidence,
        f0_hz,
    };
    sanitize_feature_vector(&mut vector);
    vector
}

fn sanitize_feature_vector(vector: &mut FeatureVector) {
    if vector.values.len() != FEATURE_VECTOR_LEN {
        vector.values.resize(FEATURE_VECTOR_LEN, 0.0);
    }
    for value in &mut vector.values {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    if !vector.rms_db.is_finite() {
        vector.rms_db = -120.0;
    }
    if !vector.high_ratio.is_finite() {
        vector.high_ratio = 0.0;
    }
    if !vector.zero_crossing_rate.is_finite() {
        vector.zero_crossing_rate = 0.0;
    }
    if !vector.spectral_flatness.is_finite() {
        vector.spectral_flatness = 0.0;
    }
    if !vector.voiced_confidence.is_finite() {
        vector.voiced_confidence = 0.0;
    }
    if !vector.f0_hz.is_finite() {
        vector.f0_hz = 0.0;
    }
}

fn prepare_frame(pcm_data: &[f32]) -> Vec<f32> {
    let mut frame = vec![0.0; FEATURE_FFT_SIZE];
    let start = pcm_data.len().saturating_sub(FEATURE_FFT_SIZE);
    let source = &pcm_data[start..];
    for (index, sample) in source.iter().take(FEATURE_FFT_SIZE).enumerate() {
        frame[index] = sample.clamp(-1.0, 1.0);
    }
    let mean = frame.iter().sum::<f32>() / FEATURE_FFT_SIZE as f32;
    for sample in &mut frame {
        *sample -= mean;
    }
    frame
}

fn power_spectrum(frame: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FEATURE_FFT_SIZE);
    let mut buffer: Vec<Complex<f32>> = frame
        .iter()
        .map(|&sample| Complex {
            re: sample,
            im: 0.0,
        })
        .collect();
    fft.process(&mut buffer);
    buffer
        .iter()
        .take(FEATURE_FFT_SIZE / 2 + 1)
        .map(|bin| bin.norm_sqr())
        .collect()
}

fn mel_like_band_energies(spectrum: &[f32], sample_rate: u32, bands: usize) -> Vec<f32> {
    let mut energies = vec![EPSILON; bands];
    let mut counts = vec![0usize; bands];
    if spectrum.is_empty() || sample_rate == 0 || bands == 0 {
        return energies;
    }

    let nyquist = sample_rate as f32 * 0.5;
    let low_mel = hz_to_mel(60.0);
    let high_mel = hz_to_mel(nyquist.min(7_600.0).max(120.0));
    let bin_hz = sample_rate as f32 / FEATURE_FFT_SIZE as f32;
    for (bin, power) in spectrum.iter().enumerate().skip(1) {
        let hz = bin as f32 * bin_hz;
        let mel = hz_to_mel(hz);
        let position = ((mel - low_mel) / (high_mel - low_mel + EPSILON)).clamp(0.0, 0.999_999);
        let band = (position * bands as f32) as usize;
        energies[band] += power.max(0.0);
        counts[band] += 1;
    }
    for (energy, count) in energies.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *energy /= *count as f32;
        }
    }
    energies
}

fn log_energies(energies: &[f32]) -> Vec<f32> {
    let mut logs: Vec<f32> = energies
        .iter()
        .map(|energy| (energy + EPSILON).ln())
        .collect();
    let mean = logs.iter().sum::<f32>() / logs.len().max(1) as f32;
    for value in &mut logs {
        *value -= mean;
    }
    logs
}

fn mfcc_without_energy(log_bands: &[f32]) -> [f32; NUM_MFCC] {
    let mut output = [0.0; NUM_MFCC];
    let band_count = log_bands.len().max(1) as f32;
    for coefficient in 1..=NUM_MFCC {
        let mut value = 0.0;
        for (band, log_energy) in log_bands.iter().enumerate() {
            let phase =
                std::f32::consts::PI * coefficient as f32 * (band as f32 + 0.5) / band_count;
            value += log_energy * phase.cos();
        }
        output[coefficient - 1] = value / band_count.sqrt();
    }
    output
}

fn delta_mfcc(current: [f32; NUM_MFCC], previous: Option<[f32; NUM_MFCC]>) -> [f32; NUM_MFCC] {
    let mut delta = [0.0; NUM_MFCC];
    if let Some(previous) = previous {
        for index in 0..NUM_MFCC {
            delta[index] = current[index] - previous[index];
        }
    }
    delta
}

fn spectral_centroid(spectrum: &[f32], sample_rate: u32) -> f32 {
    let bin_hz = sample_rate as f32 / FEATURE_FFT_SIZE as f32;
    let mut weighted = 0.0;
    let mut total = EPSILON;
    for (bin, power) in spectrum.iter().enumerate().skip(1) {
        weighted += bin as f32 * bin_hz * power.max(0.0);
        total += power.max(0.0);
    }
    weighted / total
}

fn spectral_tilt(log_bands: &[f32]) -> f32 {
    if log_bands.len() < 2 {
        return 0.0;
    }
    let midpoint = log_bands.len() / 2;
    let low = log_bands[..midpoint].iter().sum::<f32>() / midpoint as f32;
    let high = log_bands[midpoint..].iter().sum::<f32>() / (log_bands.len() - midpoint) as f32;
    low - high
}

fn spectral_flatness(energies: &[f32]) -> f32 {
    if energies.is_empty() {
        return 0.0;
    }
    let arithmetic = energies.iter().sum::<f32>() / energies.len() as f32;
    if arithmetic <= EPSILON {
        return 0.0;
    }
    let geometric = (energies
        .iter()
        .map(|energy| (energy + EPSILON).ln())
        .sum::<f32>()
        / energies.len() as f32)
        .exp();
    (geometric / arithmetic).clamp(0.0, 1.0)
}

fn estimate_f0_autocorrelation(pcm_data: &[f32], sample_rate: u32) -> (f32, f32) {
    if pcm_data.len() < 4 || sample_rate == 0 {
        return (0.0, 0.0);
    }
    let max_lag = (sample_rate / 70).max(1) as usize;
    let min_lag = (sample_rate / 500).max(1) as usize;
    let upper_lag = max_lag.min(pcm_data.len().saturating_sub(1));
    if min_lag >= upper_lag {
        return (0.0, 0.0);
    }

    let mean = pcm_data.iter().sum::<f32>() / pcm_data.len() as f32;
    let centered: Vec<f32> = pcm_data.iter().map(|sample| sample - mean).collect();
    let energy = centered.iter().map(|sample| sample * sample).sum::<f32>() + EPSILON;
    let mut best_lag = 0usize;
    let mut best_corr = 0.0;
    for lag in min_lag..=upper_lag {
        let mut corr = 0.0;
        for index in lag..centered.len() {
            corr += centered[index] * centered[index - lag];
        }
        let normalized = (corr / energy).max(0.0);
        if normalized > best_corr {
            best_corr = normalized;
            best_lag = lag;
        }
    }
    if best_lag == 0 || best_corr < 0.12 {
        return (0.0, best_corr.clamp(0.0, 1.0));
    }
    (
        sample_rate as f32 / best_lag as f32,
        best_corr.clamp(0.0, 1.0),
    )
}

fn voiced_confidence(rms: f32, zcr: f32, flatness: f32, f0_confidence: f32) -> f32 {
    let level = ((rms - 0.002) / 0.04).clamp(0.0, 1.0);
    let zcr_score = (1.0 - ((zcr - 0.04) / 0.28).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let flatness_score = (1.0 - flatness).clamp(0.0, 1.0);
    (level * (0.45 * zcr_score + 0.30 * flatness_score) + 0.25 * f0_confidence).clamp(0.0, 1.0)
}

fn calculate_rms(pcm_data: &[f32]) -> f32 {
    if pcm_data.is_empty() {
        return 0.0;
    }
    (pcm_data.iter().map(|sample| sample * sample).sum::<f32>() / pcm_data.len() as f32).sqrt()
}

fn calculate_zero_crossing_rate(pcm_data: &[f32]) -> f32 {
    if pcm_data.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for pair in pcm_data.windows(2) {
        if (pair[0] >= 0.0 && pair[1] < 0.0) || (pair[0] < 0.0 && pair[1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / (pcm_data.len() - 1) as f32
}

fn hamming_window(frame: &mut [f32]) {
    let len = frame.len();
    if len <= 1 {
        return;
    }
    for (index, sample) in frame.iter_mut().enumerate() {
        let coefficient =
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32).cos();
        *sample *= coefficient;
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz.max(0.0) / 700.0).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 44_100;

    #[test]
    fn silence_produces_low_voiced_confidence() {
        let mut extractor = FeatureExtractor::new();
        let features = extractor.extract(&vec![0.0; FEATURE_FFT_SIZE], SAMPLE_RATE);
        assert!(features.voiced_confidence < 0.35);
        assert_eq!(features.values.len(), FEATURE_VECTOR_LEN);
    }

    #[test]
    fn sustained_voiced_signal_has_higher_voiced_confidence() {
        let mut extractor = FeatureExtractor::new();
        let silence = extractor.extract(&vec![0.0; FEATURE_FFT_SIZE], SAMPLE_RATE);
        let voiced = extractor.extract(&sine(220.0, 0.4), SAMPLE_RATE);
        assert!(voiced.voiced_confidence > silence.voiced_confidence);
    }

    #[test]
    fn mfcc_vector_length_is_stable() {
        let mut extractor = FeatureExtractor::new();
        let first = extractor.extract(&sine(180.0, 0.3), SAMPLE_RATE);
        let second = extractor.extract(&sine(220.0, 0.3), SAMPLE_RATE);
        assert_eq!(first.values.len(), FEATURE_VECTOR_LEN);
        assert_eq!(second.values.len(), FEATURE_VECTOR_LEN);
    }

    #[test]
    fn features_contain_finite_numbers() {
        let mut extractor = FeatureExtractor::new();
        let features = extractor.extract(&sine(220.0, 0.4), SAMPLE_RATE);
        assert!(features.rms_db.is_finite());
        assert!(features.high_ratio.is_finite());
        assert!(features.zero_crossing_rate.is_finite());
        assert!(features.spectral_flatness.is_finite());
        assert!(features.voiced_confidence.is_finite());
        assert!(features.f0_hz.is_finite());
        assert!(features.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn amplitude_changes_rms_but_not_normalized_mfcc_shape_much() {
        let quiet = extract_features(&sine(220.0, 0.05), SAMPLE_RATE, None);
        let loud = extract_features(&sine(220.0, 0.5), SAMPLE_RATE, None);
        assert!(loud.rms_db > quiet.rms_db + 10.0);
        let mfcc_delta = quiet.values[..NUM_MFCC]
            .iter()
            .zip(loud.values[..NUM_MFCC].iter())
            .map(|(quiet, loud)| (quiet - loud).abs())
            .sum::<f32>()
            / NUM_MFCC as f32;
        assert!(mfcc_delta < 0.08, "MFCC shape drift was {mfcc_delta}");
    }

    #[test]
    fn f0_estimate_is_roughly_correct_for_sine() {
        let features = extract_features(&sine(220.0, 0.4), SAMPLE_RATE, None);
        assert!(
            (features.f0_hz - 220.0).abs() < 25.0,
            "f0 was {}",
            features.f0_hz
        );
    }

    fn sine(hz: f32, amplitude: f32) -> Vec<f32> {
        (0..FEATURE_FFT_SIZE)
            .map(|sample_index| {
                let time = sample_index as f32 / SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * hz * time).sin() * amplitude
            })
            .collect()
    }
}

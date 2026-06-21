use crate::lpc;
use num_complex::Complex;
use rustfft::FftPlanner;

pub const NUM_VOWELS: usize = 5;
pub const NUM_BANDS: usize = 16;
pub const NUM_CLASSES: usize = 9;
pub const LIPSYNC_FLAG_SINGING_MODE: u32 = 1 << 0;
pub const LIPSYNC_FLAG_TINY_NN: u32 = 1 << 1;
pub const LIPSYNC_FLAG_TIMED_CUES: u32 = 1 << 2;
pub const LIPSYNC_FLAG_ROBUST_LOUDNESS: u32 = 1 << 3;

const FRAME_SIZE: usize = 1024;
const EPSILON: f32 = 1.0e-8;
const VOWEL_CONFIDENCE_THRESHOLD: f32 = 0.34;
const SILENCE_RMS: f32 = 0.0005;
const DEFAULT_METADATA_WEIGHT: f32 = 0.55;
const DEFAULT_SMOOTHING: f32 = 0.18;
const DEFAULT_LOUDNESS_ADAPTATION: f32 = 0.07;

const BANDS_HZ: [(f32, f32); NUM_BANDS] = [
    (80.0, 200.0),
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 500.0),
    (500.0, 650.0),
    (650.0, 800.0),
    (800.0, 1000.0),
    (1000.0, 1250.0),
    (1250.0, 1500.0),
    (1500.0, 1800.0),
    (1800.0, 2200.0),
    (2200.0, 2700.0),
    (2700.0, 3300.0),
    (3300.0, 4000.0),
    (4000.0, 5000.0),
    (5000.0, 7000.0),
];

const VOWEL_PROTOTYPES: [[f32; NUM_BANDS]; NUM_VOWELS] = [
    [
        0.1, 0.2, 0.45, 0.8, 1.15, 1.35, 1.05, 0.85, 0.55, 0.25, 0.0, -0.25, -0.45, -0.65, -0.85,
        -1.0,
    ],
    [
        -0.95, -0.8, -0.65, -0.5, -0.35, -0.15, 0.05, 0.35, 0.75, 1.1, 1.35, 1.5, 1.15, 0.75, 0.25,
        -0.2,
    ],
    [
        0.65, 1.05, 1.25, 1.05, 0.8, 0.45, 0.15, -0.1, -0.3, -0.5, -0.75, -0.95, -1.1, -1.2, -1.25,
        -1.3,
    ],
    [
        -0.35, -0.15, 0.05, 0.3, 0.55, 0.75, 0.85, 0.95, 1.1, 1.15, 0.95, 0.65, 0.25, -0.15, -0.5,
        -0.8,
    ],
    [
        0.35, 0.75, 1.0, 0.95, 0.85, 0.7, 0.5, 0.3, 0.05, -0.15, -0.4, -0.65, -0.85, -1.0, -1.1,
        -1.2,
    ],
];

const TINY_NN_HIDDEN_WEIGHTS: [[f32; NUM_BANDS]; 8] = [
    [
        0.8, 0.9, 1.0, 0.85, 0.45, 0.1, -0.15, -0.35, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75,
        -0.8,
    ],
    [
        -0.6, -0.55, -0.45, -0.25, 0.05, 0.35, 0.65, 0.85, 1.0, 0.85, 0.55, 0.2, -0.1, -0.3, -0.45,
        -0.55,
    ],
    [
        -0.85, -0.8, -0.7, -0.55, -0.35, -0.1, 0.15, 0.45, 0.75, 1.0, 1.1, 1.0, 0.75, 0.45, 0.1,
        -0.25,
    ],
    [
        0.55, 0.75, 0.9, 0.95, 0.9, 0.7, 0.45, 0.15, -0.1, -0.35, -0.55, -0.7, -0.8, -0.85, -0.9,
        -0.95,
    ],
    [
        -0.25, -0.15, 0.05, 0.25, 0.45, 0.65, 0.75, 0.7, 0.55, 0.3, 0.05, -0.2, -0.4, -0.55, -0.7,
        -0.8,
    ],
    [
        -0.5, -0.45, -0.35, -0.2, 0.0, 0.25, 0.45, 0.6, 0.7, 0.65, 0.5, 0.25, 0.0, -0.2, -0.35,
        -0.45,
    ],
    [
        0.2, 0.25, 0.25, 0.15, -0.05, -0.25, -0.35, -0.2, 0.05, 0.25, 0.25, 0.05, -0.2, -0.45,
        -0.55, -0.6,
    ],
    [
        -0.2, -0.25, -0.25, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.35, 0.25, 0.1, -0.05, -0.2,
        -0.35,
    ],
];
const TINY_NN_HIDDEN_BIAS: [f32; 8] = [0.05, 0.0, 0.0, 0.05, -0.02, -0.03, -0.04, -0.05];
const TINY_NN_OUTPUT_WEIGHTS: [[f32; 8]; NUM_VOWELS] = [
    [0.15, 0.55, -0.2, 0.95, 0.55, 0.2, 0.05, -0.1],
    [-0.45, -0.2, 1.05, -0.6, -0.05, 0.65, -0.25, 0.45],
    [0.95, -0.45, -0.55, 0.55, -0.15, -0.35, 0.25, -0.25],
    [-0.25, 0.6, 0.35, -0.15, 0.95, 0.65, -0.15, 0.15],
    [0.65, -0.15, -0.55, 0.75, 0.25, -0.15, 0.45, -0.15],
];
const TINY_NN_OUTPUT_BIAS: [f32; NUM_VOWELS] = [-0.05, -0.05, -0.02, 0.0, -0.02];

#[repr(C)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Vowel {
    A = 0,
    I = 1,
    U = 2,
    E = 3,
    O = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LipSyncClass {
    Rest = 0,
    Closed = 1,
    A = 2,
    I = 3,
    U = 4,
    E = 5,
    O = 6,
    Fricative = 7,
    Other = 8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LipSyncCueKind {
    TtsViseme = 1,
    LyricTiming = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LipSyncOptions {
    pub sample_rate: u32,
    pub flags: u32,
    pub metadata_weight: f32,
    pub smoothing: f32,
    pub loudness_adaptation: f32,
}

impl Default for LipSyncOptions {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS,
            metadata_weight: DEFAULT_METADATA_WEIGHT,
            smoothing: DEFAULT_SMOOTHING,
            loudness_adaptation: DEFAULT_LOUDNESS_ADAPTATION,
        }
    }
}

impl LipSyncOptions {
    pub fn normalized(mut self) -> Self {
        self.sample_rate = self.sample_rate.max(1);
        self.metadata_weight = if self.flags & LIPSYNC_FLAG_TIMED_CUES != 0 {
            self.metadata_weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.smoothing = self.smoothing.clamp(0.0, 0.95);
        self.loudness_adaptation = self.loudness_adaptation.clamp(0.005, 0.5);
        self
    }
    pub fn singing_mode(self) -> bool {
        self.flags & LIPSYNC_FLAG_SINGING_MODE != 0
    }
    pub fn tiny_nn_enabled(self) -> bool {
        self.flags & LIPSYNC_FLAG_TINY_NN != 0
    }
    pub fn timed_cues_enabled(self) -> bool {
        self.flags & LIPSYNC_FLAG_TIMED_CUES != 0
    }
    pub fn robust_loudness_enabled(self) -> bool {
        self.flags & LIPSYNC_FLAG_ROBUST_LOUDNESS != 0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LipSyncTimedCue {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub class_index: u32,
    pub weight: f32,
    pub kind: u32,
}

impl LipSyncTimedCue {
    fn normalized(self) -> Option<Self> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || self.end_seconds <= self.start_seconds
            || self.class_index as usize >= NUM_CLASSES
        {
            return None;
        }
        Some(Self {
            start_seconds: self.start_seconds.max(0.0),
            end_seconds: self.end_seconds.max(0.0),
            class_index: self.class_index,
            weight: self.weight.clamp(0.0, 1.0),
            kind: self.kind,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VowelEvidence {
    pub scores: [f32; NUM_VOWELS],
    pub confidence: f32,
    pub f1_hz: f32,
    pub f2_hz: f32,
}

impl Default for VowelEvidence {
    fn default() -> Self {
        Self {
            scores: [0.2; NUM_VOWELS],
            confidence: 0.0,
            f1_hz: 0.0,
            f2_hz: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LipSyncFrame {
    pub posterior: [f32; NUM_CLASSES],
    pub jaw_open: f32,
    pub vowel_confidence: f32,
    pub f1_hz: f32,
    pub f2_hz: f32,
}

impl Default for LipSyncFrame {
    fn default() -> Self {
        let mut posterior = [0.0; NUM_CLASSES];
        posterior[LipSyncClass::Rest as usize] = 1.0;
        Self {
            posterior,
            jaw_open: 0.0,
            vowel_confidence: 0.0,
            f1_hz: 0.0,
            f2_hz: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LipSyncAnalyzer {
    options: LipSyncOptions,
    rolling_rms: f32,
    rolling_noise_floor: f32,
    previous_posterior: [f32; NUM_CLASSES],
    previous_jaw_open: f32,
    current_time_seconds: f32,
    timed_cues: Vec<LipSyncTimedCue>,
}

impl LipSyncAnalyzer {
    pub fn new(sample_rate: u32, singing_mode: bool) -> Self {
        let mut flags = LIPSYNC_FLAG_ROBUST_LOUDNESS;
        if singing_mode {
            flags |= LIPSYNC_FLAG_SINGING_MODE;
        }
        Self::with_options(LipSyncOptions {
            sample_rate,
            flags,
            ..LipSyncOptions::default()
        })
    }

    pub fn with_options(options: LipSyncOptions) -> Self {
        let options = options.normalized();
        Self {
            options,
            rolling_rms: 0.02,
            rolling_noise_floor: SILENCE_RMS,
            previous_posterior: LipSyncFrame::default().posterior,
            previous_jaw_open: 0.0,
            current_time_seconds: 0.0,
            timed_cues: Vec::new(),
        }
    }

    pub fn set_timed_cues(&mut self, cues: &[LipSyncTimedCue]) {
        self.timed_cues.clear();
        self.timed_cues
            .extend(cues.iter().filter_map(|cue| cue.normalized()));
        self.timed_cues.sort_by(|a, b| {
            a.start_seconds
                .partial_cmp(&b.start_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn clear_timed_cues(&mut self) {
        self.timed_cues.clear();
    }

    pub fn process(&mut self, pcm_data: &[f32]) -> LipSyncFrame {
        let time_seconds = self.current_time_seconds;
        let frame = self.process_at_time(pcm_data, time_seconds);
        self.current_time_seconds += pcm_data.len() as f32 / self.options.sample_rate as f32;
        frame
    }

    pub fn process_at_time(&mut self, pcm_data: &[f32], time_seconds: f32) -> LipSyncFrame {
        let evidence = analyze_vowel_evidence_with_options(pcm_data, self.options);
        let profile = analyze_spectral_profile(pcm_data, self.options);
        self.update_loudness_trackers(&profile);

        if profile.rms <= SILENCE_RMS && profile.peak <= SILENCE_RMS * 3.0 {
            let mut frame = LipSyncFrame {
                posterior: rest_posterior(),
                jaw_open: 0.0,
                vowel_confidence: evidence.confidence,
                f1_hz: evidence.f1_hz,
                f2_hz: evidence.f2_hz,
            };
            if self.options.timed_cues_enabled() {
                self.apply_timed_cues(&mut frame.posterior, time_seconds);
                frame.jaw_open = jaw_open_from_posterior(frame.posterior, 0.0);
            }
            self.remember_frame(&frame);
            return frame;
        }

        let activity = self.activity_score(&profile);
        let rest_score = (1.0 - activity).clamp(0.0, 0.9);
        let mut fricative_score = fricative_score(&profile) * (1.0 - rest_score);
        if profile.compression_likelihood > 0.45 {
            fricative_score *= 1.0 - 0.25 * profile.compression_likelihood;
        }

        let openness = vowel_openness(evidence.scores);
        let closed_score = ((1.0 - evidence.confidence) * (1.0 - openness) * 0.45 * activity)
            .clamp(0.0, 0.55)
            * (1.0 - rest_score)
            * (1.0 - fricative_score);
        let vowel_mass =
            (evidence.confidence * (1.0 - rest_score) * (1.0 - fricative_score) * activity)
                .clamp(0.0, 1.0);

        let mut posterior = [0.0; NUM_CLASSES];
        posterior[LipSyncClass::Rest as usize] = rest_score;
        posterior[LipSyncClass::Closed as usize] = closed_score;
        posterior[LipSyncClass::Fricative as usize] = fricative_score;
        posterior[LipSyncClass::A as usize] = evidence.scores[Vowel::A as usize] * vowel_mass;
        posterior[LipSyncClass::I as usize] = evidence.scores[Vowel::I as usize] * vowel_mass;
        posterior[LipSyncClass::U as usize] = evidence.scores[Vowel::U as usize] * vowel_mass;
        posterior[LipSyncClass::E as usize] = evidence.scores[Vowel::E as usize] * vowel_mass;
        posterior[LipSyncClass::O as usize] = evidence.scores[Vowel::O as usize] * vowel_mass;
        let used: f32 = posterior.iter().sum();
        posterior[LipSyncClass::Other as usize] = (1.0 - used).max(0.0);
        normalize_distribution(&mut posterior);

        if self.options.timed_cues_enabled() {
            self.apply_timed_cues(&mut posterior, time_seconds);
        }

        let smoothing = if self.options.singing_mode() {
            self.options.smoothing.max(0.55)
        } else {
            self.options.smoothing
        };
        if smoothing > 0.0 {
            smooth_distribution(&mut posterior, &self.previous_posterior, smoothing);
        }

        let normalized_loudness =
            (profile.rms / (self.rolling_rms + EPSILON)).clamp(0.0, 2.0) / 2.0;
        let mut jaw_open = ((0.55 * openness + 0.45 * normalized_loudness)
            * (1.0 - posterior[LipSyncClass::Rest as usize])
            * (1.0 - posterior[LipSyncClass::Closed as usize] * 0.5))
            .clamp(0.0, 1.0);
        if self.options.singing_mode() {
            jaw_open = self.previous_jaw_open * 0.45 + jaw_open * 0.55;
        }

        let frame = LipSyncFrame {
            posterior,
            jaw_open,
            vowel_confidence: evidence.confidence,
            f1_hz: evidence.f1_hz,
            f2_hz: evidence.f2_hz,
        };
        self.remember_frame(&frame);
        frame
    }

    fn remember_frame(&mut self, frame: &LipSyncFrame) {
        self.previous_posterior = frame.posterior;
        self.previous_jaw_open = frame.jaw_open;
    }

    fn update_loudness_trackers(&mut self, profile: &SpectralProfile) {
        let activity = ((profile.rms - self.rolling_noise_floor * 1.5) / 0.02).clamp(0.0, 1.0);
        let active_weight = self.options.loudness_adaptation;
        let noise_weight = if activity < 0.2 { 0.04 } else { 0.004 };
        self.rolling_rms = self.rolling_rms * (1.0 - active_weight) + profile.rms * active_weight;
        let candidate_noise = profile.rms.min(0.01);
        self.rolling_noise_floor = (self.rolling_noise_floor * (1.0 - noise_weight)
            + candidate_noise * noise_weight)
            .clamp(SILENCE_RMS, 0.02);
    }

    fn activity_score(&self, profile: &SpectralProfile) -> f32 {
        if !self.options.robust_loudness_enabled() {
            return ((profile.rms - SILENCE_RMS) / 0.025).clamp(0.0, 1.0);
        }
        let absolute_activity = ((profile.rms - SILENCE_RMS) / 0.018).clamp(0.0, 1.0);
        let adaptive_activity = ((profile.rms - self.rolling_noise_floor * 1.6)
            / (self.rolling_noise_floor * 8.0 + 0.008))
            .clamp(0.0, 1.0);
        let compression_boost = 0.18 * profile.compression_likelihood;
        absolute_activity
            .max(adaptive_activity)
            .max(compression_boost)
            .clamp(0.0, 1.0)
    }

    fn apply_timed_cues(&self, posterior: &mut [f32; NUM_CLASSES], time_seconds: f32) {
        let mut cue_distribution = [0.0; NUM_CLASSES];
        let mut cue_weight = 0.0;
        for cue in &self.timed_cues {
            if time_seconds < cue.start_seconds || time_seconds > cue.end_seconds {
                continue;
            }
            let duration = (cue.end_seconds - cue.start_seconds).max(EPSILON);
            let progress = ((time_seconds - cue.start_seconds) / duration).clamp(0.0, 1.0);
            let fade = if progress < 0.15 {
                progress / 0.15
            } else if progress > 0.85 {
                (1.0 - progress) / 0.15
            } else {
                1.0
            }
            .clamp(0.0, 1.0);
            let kind_weight = if cue.kind == LipSyncCueKind::LyricTiming as u32 {
                0.75
            } else {
                1.0
            };
            let weight = cue.weight * fade * kind_weight;
            cue_distribution[cue.class_index as usize] += weight;
            cue_weight += weight;
        }
        if cue_weight <= EPSILON {
            return;
        }
        normalize_distribution(&mut cue_distribution);
        let blend = (self.options.metadata_weight * cue_weight.clamp(0.0, 1.0)).clamp(0.0, 1.0);
        for index in 0..NUM_CLASSES {
            posterior[index] = posterior[index] * (1.0 - blend) + cue_distribution[index] * blend;
        }
        normalize_distribution(posterior);
    }
}

#[derive(Debug, Clone)]
struct SpectralProfile {
    rms: f32,
    peak: f32,
    normalized_bands: [f32; NUM_BANDS],
    high_ratio: f32,
    zero_crossing_rate: f32,
    spectral_flatness: f32,
    compression_likelihood: f32,
}

#[derive(Debug)]
pub struct Formant {
    pub frequency: f32,
    pub magnitude: f32,
}

pub fn analyze_vowel_evidence(pcm_data: &[f32], sample_rate: u32) -> VowelEvidence {
    analyze_vowel_evidence_with_options(
        pcm_data,
        LipSyncOptions {
            sample_rate,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS,
            ..LipSyncOptions::default()
        },
    )
}

pub fn analyze_vowel_evidence_with_options(
    pcm_data: &[f32],
    options: LipSyncOptions,
) -> VowelEvidence {
    let options = options.normalized();
    if pcm_data.is_empty() || options.sample_rate == 0 {
        return VowelEvidence::default();
    }

    let profile = analyze_spectral_profile(pcm_data, options);
    let (f1_hz, f2_hz) = estimate_formants(pcm_data, options.sample_rate);
    if profile.rms <= SILENCE_RMS && profile.peak <= SILENCE_RMS * 3.0 {
        return VowelEvidence {
            f1_hz,
            f2_hz,
            ..VowelEvidence::default()
        };
    }

    let mut similarities = [0.0; NUM_VOWELS];
    for vowel_index in 0..NUM_VOWELS {
        similarities[vowel_index] = cosine_similarity(
            &profile.normalized_bands,
            &normalized_prototype(VOWEL_PROTOTYPES[vowel_index]),
        );
    }

    let mut scores = softmax(similarities, 5.0);
    if options.tiny_nn_enabled() {
        let nn_scores = tiny_nn_scores(&profile);
        blend_distribution(&mut scores, &nn_scores, 0.35);
    }
    apply_compressed_voice_prior(&mut scores, profile.compression_likelihood);
    normalize_distribution(&mut scores);

    let (best, second) = best_two(scores);
    let separation = (best - second).clamp(0.0, 1.0);
    let activity = ((profile.rms - SILENCE_RMS) / 0.018)
        .clamp(0.0, 1.0)
        .max(0.2 * profile.compression_likelihood);
    let confidence =
        (activity * (0.25 + separation * 2.8) * (1.0 - profile.spectral_flatness * 0.25))
            .clamp(0.0, 1.0);

    VowelEvidence {
        scores,
        confidence,
        f1_hz,
        f2_hz,
    }
}

pub fn recognize_vowel_from_pcm(pcm_data: &[f32], sample_rate: u32) -> Option<Vowel> {
    let evidence = analyze_vowel_evidence(pcm_data, sample_rate);
    if evidence.confidence < VOWEL_CONFIDENCE_THRESHOLD {
        return None;
    }
    Some(best_vowel_from_scores(evidence.scores))
}

pub fn best_vowel_from_scores(scores: [f32; NUM_VOWELS]) -> Vowel {
    let mut best_index = 0;
    let mut best_score = scores[0];
    for (index, score) in scores.iter().enumerate().skip(1) {
        if *score > best_score {
            best_index = index;
            best_score = *score;
        }
    }
    match best_index {
        0 => Vowel::A,
        1 => Vowel::I,
        2 => Vowel::U,
        3 => Vowel::E,
        _ => Vowel::O,
    }
}

pub fn find_formants(spectrum: &[f32], sample_rate: u32, max_formants: usize) -> Vec<Formant> {
    let mut formants = Vec::new();
    let fft_size = spectrum.len();
    if fft_size < 4 || sample_rate == 0 {
        return formants;
    }
    let max_freq_index = (5000.0 / (sample_rate as f32 / fft_size as f32)) as usize;
    let upper = max_freq_index.min(fft_size / 2 - 1);
    if upper <= 1 {
        return formants;
    }
    for i in 1..upper {
        if spectrum[i].is_finite() && spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1]
        {
            let freq = i as f32 * sample_rate as f32 / fft_size as f32;
            formants.push(Formant {
                frequency: freq,
                magnitude: spectrum[i],
            });
        }
    }
    formants.truncate(max_formants);
    formants
}

fn analyze_spectral_profile(pcm_data: &[f32], options: LipSyncOptions) -> SpectralProfile {
    let raw_rms = calculate_rms(pcm_data);
    let peak = calculate_peak(pcm_data);
    let zero_crossing_rate = calculate_zero_crossing_rate(pcm_data);
    let clipping_ratio = calculate_clipping_ratio(pcm_data);
    let crest_factor = peak / (raw_rms + EPSILON);

    let mut frame = prepare_analysis_frame(pcm_data, options.robust_loudness_enabled());
    pre_emphasis(&mut frame, 0.97);
    lpc::hamming_window(&mut frame);

    let spectrum = power_spectrum(&frame);
    let band_energy = band_energies(&spectrum, options.sample_rate);
    let mut normalized_bands = normalize_log_band_energy(band_energy);
    let total_energy = band_energy.iter().sum::<f32>() + EPSILON;
    let high_energy = band_energy[12..].iter().sum::<f32>();
    let spectral_flatness = spectral_flatness(band_energy);
    let compression_likelihood =
        compression_likelihood(crest_factor, clipping_ratio, raw_rms, spectral_flatness);
    normalized_bands = compensate_compressed_feature(normalized_bands, compression_likelihood);

    SpectralProfile {
        rms: raw_rms,
        peak,
        normalized_bands,
        high_ratio: high_energy / total_energy,
        zero_crossing_rate,
        spectral_flatness,
        compression_likelihood,
    }
}

fn prepare_analysis_frame(pcm_data: &[f32], robust_loudness: bool) -> Vec<f32> {
    let mut frame = vec![0.0; FRAME_SIZE];
    let start = pcm_data.len().saturating_sub(FRAME_SIZE);
    let source = &pcm_data[start..];
    for (index, sample) in source.iter().take(FRAME_SIZE).enumerate() {
        frame[index] = sample.clamp(-1.0, 1.0);
    }
    if !robust_loudness {
        return frame;
    }

    let mean = frame.iter().sum::<f32>() / FRAME_SIZE as f32;
    for sample in &mut frame {
        *sample -= mean;
    }
    let rms = calculate_rms(&frame);
    if rms > EPSILON {
        let gain = (0.18 / rms).clamp(0.2, 12.0);
        for sample in &mut frame {
            *sample = soft_limit(*sample * gain);
        }
    }
    frame
}

fn power_spectrum(frame: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);
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
        .take(FRAME_SIZE / 2 + 1)
        .map(|bin| bin.norm_sqr())
        .collect()
}

fn band_energies(spectrum: &[f32], sample_rate: u32) -> [f32; NUM_BANDS] {
    let mut energies = [EPSILON; NUM_BANDS];
    let mut counts = [0usize; NUM_BANDS];
    if sample_rate == 0 || spectrum.is_empty() {
        return energies;
    }
    let bin_hz = sample_rate as f32 / FRAME_SIZE as f32;
    for (bin, power) in spectrum.iter().enumerate().skip(1) {
        let hz = bin as f32 * bin_hz;
        for (band_index, (low_hz, high_hz)) in BANDS_HZ.iter().enumerate() {
            if hz >= *low_hz && hz < *high_hz {
                energies[band_index] += power.max(0.0);
                counts[band_index] += 1;
                break;
            }
        }
    }
    for (energy, count) in energies.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *energy /= *count as f32;
        }
    }
    energies
}

fn normalize_log_band_energy(band_energy: [f32; NUM_BANDS]) -> [f32; NUM_BANDS] {
    let mut feature = [0.0; NUM_BANDS];
    for (index, energy) in band_energy.iter().enumerate() {
        feature[index] = (energy + EPSILON).ln();
    }
    normalize_feature(feature)
}

fn normalize_feature(mut feature: [f32; NUM_BANDS]) -> [f32; NUM_BANDS] {
    let mean = feature.iter().sum::<f32>() / NUM_BANDS as f32;
    for value in &mut feature {
        *value -= mean;
    }
    let norm = feature
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm <= EPSILON {
        return [0.0; NUM_BANDS];
    }
    for value in &mut feature {
        *value /= norm + EPSILON;
    }
    feature
}

fn compensate_compressed_feature(
    feature: [f32; NUM_BANDS],
    compression_likelihood: f32,
) -> [f32; NUM_BANDS] {
    let amount = compression_likelihood.clamp(0.0, 1.0);
    if amount <= 0.01 {
        return feature;
    }

    let mut compensated = feature;
    for index in 0..NUM_BANDS {
        let left = if index > 0 {
            feature[index - 1]
        } else {
            feature[index]
        };
        let right = if index + 1 < NUM_BANDS {
            feature[index + 1]
        } else {
            feature[index]
        };
        compensated[index] =
            feature[index] * (1.0 - 0.35 * amount) + (left + right) * 0.175 * amount;
    }

    for index in 10..NUM_BANDS {
        let high_shelf = (index - 9) as f32 / (NUM_BANDS - 9) as f32;
        compensated[index] -= amount * high_shelf * 0.28;
    }

    normalize_feature(compensated)
}
fn normalized_prototype(prototype: [f32; NUM_BANDS]) -> [f32; NUM_BANDS] {
    normalize_feature(prototype)
}
fn cosine_similarity(a: &[f32; NUM_BANDS], b: &[f32; NUM_BANDS]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

fn apply_compressed_voice_prior(scores: &mut [f32; NUM_VOWELS], compression_likelihood: f32) {
    let weight = (compression_likelihood * 1.8).clamp(0.0, 0.95);
    if weight <= 0.01 {
        return;
    }

    let compressed_prior = [0.50, 0.03, 0.10, 0.17, 0.20];
    for index in 0..NUM_VOWELS {
        scores[index] = scores[index] * (1.0 - weight) + compressed_prior[index] * weight;
    }
    normalize_distribution(scores);
}
fn tiny_nn_scores(profile: &SpectralProfile) -> [f32; NUM_VOWELS] {
    let mut hidden = [0.0; 8];
    for hidden_index in 0..8 {
        let dot = cosine_like_dot(
            &profile.normalized_bands,
            &TINY_NN_HIDDEN_WEIGHTS[hidden_index],
        ) + TINY_NN_HIDDEN_BIAS[hidden_index];
        hidden[hidden_index] = dot.max(0.0);
    }
    let mut logits = [0.0; NUM_VOWELS];
    for vowel in 0..NUM_VOWELS {
        logits[vowel] = TINY_NN_OUTPUT_BIAS[vowel];
        for hidden_index in 0..8 {
            logits[vowel] += hidden[hidden_index] * TINY_NN_OUTPUT_WEIGHTS[vowel][hidden_index];
        }
    }
    softmax(logits, 3.0)
}

fn cosine_like_dot(feature: &[f32; NUM_BANDS], weights: &[f32; NUM_BANDS]) -> f32 {
    let normalized_weights = normalize_feature(*weights);
    cosine_similarity(feature, &normalized_weights)
}

fn softmax<const N: usize>(values: [f32; N], temperature: f32) -> [f32; N] {
    let mut max_value = f32::NEG_INFINITY;
    for value in values {
        max_value = max_value.max(value * temperature);
    }
    let mut output = [0.0; N];
    let mut sum = 0.0;
    for index in 0..N {
        output[index] = ((values[index] * temperature) - max_value).exp();
        sum += output[index];
    }
    if sum <= EPSILON {
        return [1.0 / N as f32; N];
    }
    for value in &mut output {
        *value /= sum;
    }
    output
}

fn best_two(values: [f32; NUM_VOWELS]) -> (f32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in values {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    (best, second)
}

fn clamp_scores(scores: &mut [f32]) {
    for score in scores {
        *score = score.clamp(0.0, 1.0);
    }
}

fn normalize_distribution(scores: &mut [f32]) {
    clamp_scores(scores);
    let sum = scores.iter().sum::<f32>();
    if sum <= EPSILON {
        let uniform = 1.0 / scores.len() as f32;
        for score in scores {
            *score = uniform;
        }
        return;
    }
    for score in scores {
        *score /= sum;
    }
}

fn blend_distribution<const N: usize>(base: &mut [f32; N], overlay: &[f32; N], weight: f32) {
    let weight = weight.clamp(0.0, 1.0);
    for index in 0..N {
        base[index] = base[index] * (1.0 - weight) + overlay[index] * weight;
    }
    normalize_distribution(base);
}

fn smooth_distribution(
    scores: &mut [f32; NUM_CLASSES],
    previous: &[f32; NUM_CLASSES],
    previous_weight: f32,
) {
    let current_weight = 1.0 - previous_weight;
    for index in 0..NUM_CLASSES {
        scores[index] = previous[index] * previous_weight + scores[index] * current_weight;
    }
    normalize_distribution(scores);
}

fn rest_posterior() -> [f32; NUM_CLASSES] {
    let mut posterior = [0.0; NUM_CLASSES];
    posterior[LipSyncClass::Rest as usize] = 0.98;
    posterior[LipSyncClass::Other as usize] = 0.02;
    posterior
}

fn fricative_score(profile: &SpectralProfile) -> f32 {
    let high_component = ((profile.high_ratio - 0.42) / 0.28).clamp(0.0, 1.0);
    let zcr_component = ((profile.zero_crossing_rate - 0.18) / 0.25).clamp(0.0, 1.0);
    let flat_component = ((profile.spectral_flatness - 0.45) / 0.35).clamp(0.0, 1.0);
    (0.58 * high_component + 0.27 * zcr_component + 0.15 * flat_component).clamp(0.0, 1.0)
}

fn vowel_openness(scores: [f32; NUM_VOWELS]) -> f32 {
    (scores[Vowel::A as usize] * 1.0
        + scores[Vowel::I as usize] * 0.22
        + scores[Vowel::U as usize] * 0.32
        + scores[Vowel::E as usize] * 0.62
        + scores[Vowel::O as usize] * 0.55)
        .clamp(0.0, 1.0)
}

fn jaw_open_from_posterior(posterior: [f32; NUM_CLASSES], normalized_loudness: f32) -> f32 {
    let vowel_open = posterior[LipSyncClass::A as usize] * 1.0
        + posterior[LipSyncClass::I as usize] * 0.22
        + posterior[LipSyncClass::U as usize] * 0.32
        + posterior[LipSyncClass::E as usize] * 0.62
        + posterior[LipSyncClass::O as usize] * 0.55;
    ((0.65 * vowel_open + 0.35 * normalized_loudness)
        * (1.0 - posterior[LipSyncClass::Rest as usize])
        * (1.0 - posterior[LipSyncClass::Closed as usize] * 0.5))
        .clamp(0.0, 1.0)
}

fn calculate_rms(pcm_data: &[f32]) -> f32 {
    if pcm_data.is_empty() {
        return 0.0;
    }
    (pcm_data.iter().map(|sample| sample * sample).sum::<f32>() / pcm_data.len() as f32).sqrt()
}
fn calculate_peak(pcm_data: &[f32]) -> f32 {
    pcm_data
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max)
}
fn calculate_clipping_ratio(pcm_data: &[f32]) -> f32 {
    if pcm_data.is_empty() {
        return 0.0;
    }
    pcm_data
        .iter()
        .filter(|sample| sample.abs() >= 0.985)
        .count() as f32
        / pcm_data.len() as f32
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

fn spectral_flatness(band_energy: [f32; NUM_BANDS]) -> f32 {
    let arithmetic = band_energy.iter().sum::<f32>() / NUM_BANDS as f32;
    if arithmetic <= EPSILON {
        return 0.0;
    }
    let geometric = (band_energy
        .iter()
        .map(|value| (value + EPSILON).ln())
        .sum::<f32>()
        / NUM_BANDS as f32)
        .exp();
    (geometric / arithmetic).clamp(0.0, 1.0)
}

fn compression_likelihood(crest_factor: f32, clipping_ratio: f32, rms: f32, flatness: f32) -> f32 {
    let low_crest = ((3.0 - crest_factor) / 2.0).clamp(0.0, 1.0);
    let clipped = (clipping_ratio / 0.08).clamp(0.0, 1.0);
    let loud = ((rms - 0.25) / 0.45).clamp(0.0, 1.0);
    (0.45 * low_crest + 0.35 * clipped + 0.15 * loud + 0.05 * flatness).clamp(0.0, 1.0)
}

fn soft_limit(value: f32) -> f32 {
    (value * 0.85).tanh()
}

fn pre_emphasis(signal: &mut [f32], coefficient: f32) {
    if signal.len() < 2 {
        return;
    }
    for i in (1..signal.len()).rev() {
        signal[i] -= coefficient * signal[i - 1];
    }
}

fn estimate_formants(pcm_data: &[f32], sample_rate: u32) -> (f32, f32) {
    if pcm_data.is_empty() || sample_rate == 0 || calculate_rms(pcm_data) <= SILENCE_RMS {
        return (0.0, 0.0);
    }
    let mut frame = prepare_analysis_frame(pcm_data, true);
    pre_emphasis(&mut frame, 0.97);
    lpc::hamming_window(&mut frame);
    let mut acf = lpc::autocorrelate(&frame);
    let acf0 = acf[0];
    if acf0 <= EPSILON {
        return (0.0, 0.0);
    }
    for value in &mut acf {
        *value /= acf0;
    }
    let order = ((sample_rate / 1000) as usize + 4).clamp(8, 64);
    if let Some((alpha, err)) = lpc::levinson_durbin(&acf, order) {
        let gain = err * acf0;
        let spectral_envelope = lpc::lpc_to_spectral_envelope(&alpha, gain, FRAME_SIZE);
        let formants = find_formants(&spectral_envelope, sample_rate, 2);
        let f1 = formants
            .get(0)
            .map(|formant| formant.frequency)
            .unwrap_or(0.0);
        let f2 = formants
            .get(1)
            .map(|formant| formant.frequency)
            .unwrap_or(0.0);
        return (f1, f2);
    }
    (0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 44_100;

    #[test]
    fn evidence_is_low_confidence_for_silence() {
        let pcm = vec![0.0; FRAME_SIZE];
        let evidence = analyze_vowel_evidence(&pcm, SAMPLE_RATE);
        assert!(evidence.confidence < 0.05);
        assert_scores_normalized(evidence.scores);
    }

    #[test]
    fn vowel_scores_are_stable_across_loudness() {
        let quiet =
            synthetic_vowel_like_signal(0.04, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let loud = synthetic_vowel_like_signal(0.8, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let quiet_scores = analyze_vowel_evidence(&quiet, SAMPLE_RATE).scores;
        let loud_scores = analyze_vowel_evidence(&loud, SAMPLE_RATE).scores;
        let difference = quiet_scores
            .iter()
            .zip(loud_scores.iter())
            .map(|(quiet, loud)| (quiet - loud).abs())
            .sum::<f32>();
        assert!(difference < 0.18, "score drift was {difference}");
    }

    #[test]
    fn clipped_or_compressed_voice_keeps_vowel_shape() {
        let clean =
            synthetic_vowel_like_signal(0.35, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let clipped: Vec<f32> = clean
            .iter()
            .map(|sample| (*sample * 5.0).clamp(-0.92, 0.92))
            .collect();
        let clean_scores = analyze_vowel_evidence(&clean, SAMPLE_RATE).scores;
        let clipped_scores = analyze_vowel_evidence(&clipped, SAMPLE_RATE).scores;
        let difference = clean_scores
            .iter()
            .zip(clipped_scores.iter())
            .map(|(clean, clipped)| (clean - clipped).abs())
            .sum::<f32>();
        assert!(difference < 0.3, "compressed score drift was {difference}");
    }

    #[test]
    fn tiny_nn_option_changes_but_normalizes_scores() {
        let pcm = synthetic_vowel_like_signal(0.4, &[(180.0, 0.3), (950.0, 1.0), (2200.0, 0.6)]);
        let base = analyze_vowel_evidence(&pcm, SAMPLE_RATE).scores;
        let tiny = analyze_vowel_evidence_with_options(
            &pcm,
            LipSyncOptions {
                sample_rate: SAMPLE_RATE,
                flags: LIPSYNC_FLAG_ROBUST_LOUDNESS | LIPSYNC_FLAG_TINY_NN,
                ..LipSyncOptions::default()
            },
        )
        .scores;
        let difference = base
            .iter()
            .zip(tiny.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert_scores_normalized(tiny);
        assert!(difference > 0.001);
    }

    #[test]
    fn recognize_vowel_uses_evidence_wrapper_without_panicking() {
        let pcm = synthetic_vowel_like_signal(0.5, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let result = recognize_vowel_from_pcm(&pcm, SAMPLE_RATE);
        assert!(result.is_some());
    }

    #[test]
    fn analyzer_returns_rest_and_closed_classes_for_zero_pcm() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let frame = analyzer.process(&vec![0.0; FRAME_SIZE]);
        assert!(frame.posterior[LipSyncClass::Rest as usize] > 0.9);
        assert!(frame.posterior[LipSyncClass::Closed as usize] <= 0.05);
        assert!(frame.jaw_open <= 0.01);
        assert_scores_normalized(frame.posterior);
    }

    #[test]
    fn high_frequency_noise_prefers_fricative_over_vowels() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let noise = synthetic_high_frequency_noise(0.35);
        let frame = analyzer.process(&noise);
        let best_vowel = frame.posterior[LipSyncClass::A as usize..=LipSyncClass::O as usize]
            .iter()
            .copied()
            .fold(0.0, f32::max);
        assert!(frame.posterior[LipSyncClass::Fricative as usize] > best_vowel);
    }

    #[test]
    fn voiced_signal_prefers_a_vowel_class() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let voiced =
            synthetic_vowel_like_signal(0.5, &[(120.0, 0.4), (720.0, 1.0), (1150.0, 0.65)]);
        let frame = analyzer.process(&voiced);
        let best_vowel = frame.posterior[LipSyncClass::A as usize..=LipSyncClass::O as usize]
            .iter()
            .copied()
            .fold(0.0, f32::max);
        assert!(best_vowel > frame.posterior[LipSyncClass::Rest as usize]);
        assert!(best_vowel > frame.posterior[LipSyncClass::Fricative as usize]);
        assert!(frame.jaw_open > 0.05);
    }

    #[test]
    fn singing_mode_smooths_fast_vowel_switches() {
        let a_like =
            synthetic_vowel_like_signal(0.45, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let i_like =
            synthetic_vowel_like_signal(0.45, &[(180.0, 0.25), (2200.0, 1.0), (3100.0, 0.6)]);
        let mut normal = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let mut singing = LipSyncAnalyzer::new(SAMPLE_RATE, true);
        normal.process(&a_like);
        singing.process(&a_like);
        let normal_i = normal.process(&i_like).posterior[LipSyncClass::I as usize];
        let singing_i = singing.process(&i_like).posterior[LipSyncClass::I as usize];
        assert!(singing_i < normal_i);
    }

    #[test]
    fn timed_tts_cue_biases_posterior_at_time() {
        let mut analyzer = LipSyncAnalyzer::with_options(LipSyncOptions {
            sample_rate: SAMPLE_RATE,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS | LIPSYNC_FLAG_TIMED_CUES,
            metadata_weight: 0.8,
            ..LipSyncOptions::default()
        });
        analyzer.set_timed_cues(&[LipSyncTimedCue {
            start_seconds: 0.1,
            end_seconds: 0.4,
            class_index: LipSyncClass::O as u32,
            weight: 1.0,
            kind: LipSyncCueKind::TtsViseme as u32,
        }]);
        let pcm = synthetic_vowel_like_signal(0.25, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let frame = analyzer.process_at_time(&pcm, 0.2);
        assert!(
            frame.posterior[LipSyncClass::O as usize] > frame.posterior[LipSyncClass::A as usize]
        );
    }

    fn synthetic_vowel_like_signal(amplitude: f32, partials: &[(f32, f32)]) -> Vec<f32> {
        (0..FRAME_SIZE)
            .map(|sample_index| {
                let time = sample_index as f32 / SAMPLE_RATE as f32;
                let sum = partials
                    .iter()
                    .map(|(hz, weight)| (2.0 * std::f32::consts::PI * hz * time).sin() * weight)
                    .sum::<f32>();
                (sum * amplitude / partials.len() as f32).clamp(-1.0, 1.0)
            })
            .collect()
    }

    fn synthetic_high_frequency_noise(amplitude: f32) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        (0..FRAME_SIZE)
            .map(|sample_index| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let random = ((seed >> 8) as f32 / 0x00ff_ffff as f32) * 2.0 - 1.0;
                let time = sample_index as f32 / SAMPLE_RATE as f32;
                let carrier = (2.0 * std::f32::consts::PI * 5_200.0 * time).sin();
                (random * carrier * amplitude).clamp(-1.0, 1.0)
            })
            .collect()
    }

    fn assert_scores_normalized<const N: usize>(scores: [f32; N]) {
        for score in scores {
            assert!((0.0..=1.0).contains(&score));
        }
        let sum = scores.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.001, "score sum was {sum}");
    }
}

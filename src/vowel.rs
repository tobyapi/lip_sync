use crate::{gmm, lpc, normalization::RollingLoudness};
use num_complex::Complex;
use rustfft::FftPlanner;

pub const NUM_VOWELS: usize = 5;
pub const NUM_BANDS: usize = 16;
pub const NUM_CLASSES: usize = 9;
pub const LIPSYNC_FLAG_SINGING_MODE: u32 = 1 << 0;
pub const LIPSYNC_FLAG_TINY_NN: u32 = 1 << 1;
pub const LIPSYNC_FLAG_TIMED_CUES: u32 = 1 << 2;
pub const LIPSYNC_FLAG_ROBUST_LOUDNESS: u32 = 1 << 3;
pub const LIPSYNC_FLAG_GMM: u32 = 1 << 4;

const FRAME_SIZE: usize = 1024;
const EPSILON: f32 = 1.0e-8;
const SILENCE_RMS: f32 = 0.0005;
const DEFAULT_METADATA_WEIGHT: f32 = 0.55;
const DEFAULT_SMOOTHING: f32 = 0.18;
const DEFAULT_LOUDNESS_ADAPTATION: f32 = 0.07;
const NORMAL_WINDOW_MS: f32 = 25.0;
const SINGING_WINDOW_MS: f32 = 40.0;
const ANALYSIS_HOP_MS: f32 = 10.0;
const COMPRESSED_VOWEL_PRIOR_MAX_WEIGHT: f32 = 0.18;

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

const NUM_VOWEL_PROTOTYPES: usize = 8;
const MULTI_PROTOTYPE_TEMPERATURE: f32 = 6.0;
const VOWEL_PROTOTYPES: [[[f32; NUM_BANDS]; NUM_VOWEL_PROTOTYPES]; NUM_VOWELS] = [
    [
        [
            0.1, 0.2, 0.45, 0.8, 1.15, 1.35, 1.05, 0.85, 0.55, 0.25, 0.0, -0.25, -0.45, -0.65,
            -0.85, -1.0,
        ],
        [
            0.26, 0.34, 0.57, 0.9, 1.21, 1.39, 1.07, 0.85, 0.53, 0.21, -0.06, -0.33, -0.55, -0.77,
            -0.99, -1.16,
        ],
        [
            -0.06, 0.06, 0.33, 0.7, 1.09, 1.31, 1.03, 0.85, 0.57, 0.29, 0.06, -0.17, -0.35, -0.53,
            -0.71, -0.84,
        ],
        [
            0.146, 0.222, 0.427, 0.718, 1.009, 1.171, 0.903, 0.721, 0.453, 0.185, -0.04, -0.265,
            -0.437, -0.609, -0.781, -0.91,
        ],
        [
            0.116, 0.244, 0.506, 0.86, 1.206, 1.37, 1.146, 0.894, 0.59, 0.278, 0.0, -0.262, -0.49,
            -0.71, -0.922, -1.076,
        ],
        [
            0.22, 0.3, 0.53, 0.86, 1.19, 1.37, 1.05, 0.83, 0.51, 0.19, -0.08, -0.35, -0.57, -0.79,
            -1.01, -1.18,
        ],
        [
            -0.02, 0.1, 0.37, 0.74, 1.11, 1.33, 1.05, 0.87, 0.59, 0.31, 0.08, -0.15, -0.33, -0.51,
            -0.69, -0.82,
        ],
        [
            0.134, 0.268, 0.543, 0.832, 1.121, 1.269, 0.967, 0.769, 0.497, 0.235, 0.02, -0.205,
            -0.403, -0.611, -0.819, -0.98,
        ],
    ],
    [
        [
            -0.95, -0.8, -0.65, -0.5, -0.35, -0.15, 0.05, 0.35, 0.75, 1.1, 1.35, 1.5, 1.15, 0.75,
            0.25, -0.2,
        ],
        [
            -0.79, -0.66, -0.53, -0.4, -0.29, -0.11, 0.07, 0.35, 0.73, 1.06, 1.29, 1.42, 1.05,
            0.63, 0.11, -0.36,
        ],
        [
            -1.11, -0.94, -0.77, -0.6, -0.41, -0.19, 0.03, 0.35, 0.77, 1.14, 1.41, 1.58, 1.25,
            0.87, 0.39, -0.04,
        ],
        [
            -0.757, -0.638, -0.519, -0.4, -0.281, -0.119, 0.043, 0.291, 0.625, 0.916, 1.121, 1.24,
            0.939, 0.595, 0.165, -0.222,
        ],
        [
            -0.926, -0.78, -0.61, -0.44, -0.262, -0.05, 0.146, 0.426, 0.782, 1.104, 1.334, 1.4,
            1.102, 0.674, 0.178, -0.228,
        ],
        [
            -0.83, -0.7, -0.57, -0.44, -0.31, -0.13, 0.05, 0.33, 0.71, 1.04, 1.27, 1.4, 1.03, 0.61,
            0.09, -0.38,
        ],
        [
            -1.07, -0.9, -0.73, -0.56, -0.39, -0.17, 0.05, 0.37, 0.79, 1.16, 1.43, 1.6, 1.27, 0.89,
            0.41, -0.02,
        ],
        [
            -0.853, -0.672, -0.491, -0.39, -0.289, -0.141, 0.027, 0.299, 0.685, 1.034, 1.289, 1.44,
            1.101, 0.705, 0.215, -0.228,
        ],
    ],
    [
        [
            0.65, 1.05, 1.25, 1.05, 0.8, 0.45, 0.15, -0.1, -0.3, -0.5, -0.75, -0.95, -1.1, -1.2,
            -1.25, -1.3,
        ],
        [
            0.81, 1.19, 1.37, 1.15, 0.86, 0.49, 0.17, -0.1, -0.32, -0.54, -0.81, -1.03, -1.2,
            -1.32, -1.39, -1.46,
        ],
        [
            0.49, 0.91, 1.13, 0.95, 0.74, 0.41, 0.13, -0.1, -0.28, -0.46, -0.69, -0.87, -1.0,
            -1.08, -1.11, -1.14,
        ],
        [
            0.619, 0.953, 1.115, 0.933, 0.708, 0.397, 0.129, -0.096, -0.278, -0.46, -0.685, -0.867,
            -0.996, -1.082, -1.125, -1.168,
        ],
        [
            0.714, 1.038, 1.226, 1.102, 0.864, 0.558, 0.238, -0.032, -0.26, -0.488, -0.742, -0.962,
            -1.132, -1.252, -1.33, -1.392,
        ],
        [
            0.77, 1.15, 1.33, 1.11, 0.84, 0.47, 0.15, -0.12, -0.34, -0.56, -0.83, -1.05, -1.22,
            -1.34, -1.41, -1.48,
        ],
        [
            0.53, 0.95, 1.17, 0.99, 0.76, 0.43, 0.15, -0.08, -0.26, -0.44, -0.67, -0.85, -0.98,
            -1.06, -1.09, -1.12,
        ],
        [
            0.651, 1.067, 1.295, 1.067, 0.792, 0.423, 0.121, -0.124, -0.302, -0.47, -0.685, -0.863,
            -1.014, -1.128, -1.195, -1.262,
        ],
    ],
    [
        [
            -0.35, -0.15, 0.05, 0.3, 0.55, 0.75, 0.85, 0.95, 1.1, 1.15, 0.95, 0.65, 0.25, -0.15,
            -0.5, -0.8,
        ],
        [
            -0.19, -0.01, 0.17, 0.4, 0.61, 0.79, 0.87, 0.95, 1.08, 1.11, 0.89, 0.57, 0.15, -0.27,
            -0.64, -0.96,
        ],
        [
            -0.51, -0.29, -0.07, 0.2, 0.49, 0.71, 0.83, 0.95, 1.12, 1.19, 1.01, 0.73, 0.35, -0.03,
            -0.36, -0.64,
        ],
        [
            -0.241, -0.079, 0.083, 0.288, 0.493, 0.655, 0.731, 0.807, 0.926, 0.959, 0.777, 0.509,
            0.165, -0.179, -0.48, -0.738,
        ],
        [
            -0.318, -0.13, 0.098, 0.36, 0.622, 0.834, 0.93, 1.018, 1.124, 1.13, 0.934, 0.614, 0.21,
            -0.202, -0.572, -0.852,
        ],
        [
            -0.23, -0.05, 0.13, 0.36, 0.59, 0.77, 0.85, 0.93, 1.06, 1.09, 0.87, 0.55, 0.13, -0.29,
            -0.66, -0.98,
        ],
        [
            -0.47, -0.25, -0.03, 0.24, 0.51, 0.73, 0.85, 0.97, 1.14, 1.21, 1.03, 0.75, 0.37, -0.01,
            -0.34, -0.62,
        ],
        [
            -0.289, -0.061, 0.167, 0.362, 0.557, 0.705, 0.779, 0.863, 1.014, 1.081, 0.913, 0.641,
            0.255, -0.141, -0.49, -0.792,
        ],
    ],
    [
        [
            0.35, 0.75, 1.0, 0.95, 0.85, 0.7, 0.5, 0.3, 0.05, -0.15, -0.4, -0.65, -0.85, -1.0,
            -1.1, -1.2,
        ],
        [
            0.51, 0.89, 1.12, 1.05, 0.91, 0.74, 0.52, 0.3, 0.03, -0.19, -0.46, -0.73, -0.95, -1.12,
            -1.24, -1.36,
        ],
        [
            0.19, 0.61, 0.88, 0.85, 0.79, 0.66, 0.48, 0.3, 0.07, -0.11, -0.34, -0.57, -0.75, -0.88,
            -0.96, -1.04,
        ],
        [
            0.361, 0.695, 0.9, 0.847, 0.751, 0.612, 0.43, 0.248, 0.023, -0.159, -0.384, -0.609,
            -0.781, -0.91, -0.996, -1.082,
        ],
        [
            0.414, 0.746, 0.992, 1.002, 0.922, 0.792, 0.58, 0.352, 0.098, -0.138, -0.4, -0.662,
            -0.882, -1.052, -1.18, -1.284,
        ],
        [
            0.47, 0.85, 1.08, 1.01, 0.89, 0.72, 0.5, 0.28, 0.01, -0.21, -0.48, -0.75, -0.97, -1.14,
            -1.26, -1.38,
        ],
        [
            0.23, 0.65, 0.92, 0.89, 0.81, 0.68, 0.5, 0.32, 0.09, -0.09, -0.32, -0.55, -0.73, -0.86,
            -0.94, -1.02,
        ],
        [
            0.369, 0.785, 1.06, 0.973, 0.839, 0.658, 0.45, 0.252, 0.027, -0.141, -0.356, -0.581,
            -0.779, -0.94, -1.054, -1.168,
        ],
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            ..Self::default()
        }
    }

    pub fn singing_preset(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS | LIPSYNC_FLAG_SINGING_MODE,
            metadata_weight: DEFAULT_METADATA_WEIGHT,
            smoothing: 0.65,
            loudness_adaptation: DEFAULT_LOUDNESS_ADAPTATION,
        }
    }

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
    pub fn gmm_enabled(self) -> bool {
        self.flags & LIPSYNC_FLAG_GMM != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VowelClassifierKind {
    MultiPrototype,
    DiagonalGmm,
}

impl VowelClassifierKind {
    fn from_options(options: LipSyncOptions) -> Self {
        if options.gmm_enabled() {
            Self::DiagonalGmm
        } else {
            Self::MultiPrototype
        }
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

impl LipSyncFrame {
    pub fn best_class(&self) -> LipSyncClass {
        best_class(&self.posterior)
    }

    pub fn best_class_score(&self) -> f32 {
        self.posterior[self.best_class() as usize]
    }
}

pub fn lip_sync_class_from_index(index: u32) -> Option<LipSyncClass> {
    match index {
        0 => Some(LipSyncClass::Rest),
        1 => Some(LipSyncClass::Closed),
        2 => Some(LipSyncClass::A),
        3 => Some(LipSyncClass::I),
        4 => Some(LipSyncClass::U),
        5 => Some(LipSyncClass::E),
        6 => Some(LipSyncClass::O),
        7 => Some(LipSyncClass::Fricative),
        8 => Some(LipSyncClass::Other),
        _ => None,
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LipSyncDebugFrame {
    pub frame: LipSyncFrame,
    pub vowel_scores: [f32; NUM_VOWELS],
    pub activity: f32,
    pub rms: f32,
    pub high_ratio: f32,
    pub zero_crossing_rate: f32,
    pub spectral_flatness: f32,
    pub compression_likelihood: f32,
    pub raw_best_vowel: i32,
}

impl Default for LipSyncDebugFrame {
    fn default() -> Self {
        Self {
            frame: LipSyncFrame::default(),
            vowel_scores: [0.2; NUM_VOWELS],
            activity: 0.0,
            rms: 0.0,
            high_ratio: 0.0,
            zero_crossing_rate: 0.0,
            spectral_flatness: 0.0,
            compression_likelihood: 0.0,
            raw_best_vowel: -1,
        }
    }
}

impl LipSyncDebugFrame {
    fn from_parts(
        frame: LipSyncFrame,
        evidence: VowelEvidence,
        profile: &SpectralProfile,
        activity: f32,
    ) -> Self {
        Self {
            frame,
            vowel_scores: evidence.scores,
            activity,
            rms: profile.rms,
            high_ratio: profile.high_ratio,
            zero_crossing_rate: profile.zero_crossing_rate,
            spectral_flatness: profile.spectral_flatness,
            compression_likelihood: profile.compression_likelihood,
            raw_best_vowel: if evidence.confidence > EPSILON {
                best_vowel_from_scores(evidence.scores) as i32
            } else {
                -1
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AnalysisRing {
    audio: Vec<f32>,
    total_samples_seen: usize,
    next_analysis_end_sample: usize,
    previous_window_end_sample: usize,
    last_window_end_sample: usize,
}

impl AnalysisRing {
    fn push(&mut self, pcm_data: &[f32], capacity: usize) {
        self.audio
            .extend(pcm_data.iter().map(|sample| sample.clamp(-1.0, 1.0)));
        self.total_samples_seen = self.total_samples_seen.saturating_add(pcm_data.len());
        if self.audio.len() > capacity {
            let overflow = self.audio.len() - capacity;
            self.audio.drain(0..overflow);
        }
    }

    fn has_enough_samples(&self, window_samples: usize) -> bool {
        self.total_samples_seen >= window_samples
    }

    fn next_due_window(&mut self, window_samples: usize, hop_samples: usize) -> Option<Vec<f32>> {
        if !self.has_enough_samples(window_samples) {
            return None;
        }
        if self.next_analysis_end_sample < window_samples {
            self.next_analysis_end_sample = window_samples;
        }

        while self.next_analysis_end_sample <= self.total_samples_seen {
            let end_sample = self.next_analysis_end_sample;
            self.next_analysis_end_sample =
                self.next_analysis_end_sample.saturating_add(hop_samples);
            if let Some(window) = self.window_ending_at(end_sample, window_samples) {
                self.previous_window_end_sample = self.last_window_end_sample;
                self.last_window_end_sample = end_sample;
                return Some(window);
            }
        }
        None
    }

    fn window_ending_at(&self, end_sample: usize, window_samples: usize) -> Option<Vec<f32>> {
        if end_sample < window_samples || end_sample > self.total_samples_seen {
            return None;
        }
        let ring_start_sample = self.total_samples_seen.saturating_sub(self.audio.len());
        if end_sample < ring_start_sample {
            return None;
        }
        let end_index = end_sample - ring_start_sample;
        let start_index = end_index.checked_sub(window_samples)?;
        if end_index > self.audio.len() {
            return None;
        }
        Some(self.audio[start_index..end_index].to_vec())
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.audio.len()
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
    rolling_loudness: RollingLoudness,
    closed_detector: ClosedDetector,
    current_class: LipSyncClass,
    hold_time_seconds: f32,
    previous_time_step_seconds: f32,
    last_switch_confidence: f32,
    stream: AnalysisRing,
    latest_frame: LipSyncFrame,
    latest_debug_frame: LipSyncDebugFrame,
    analysis_count: usize,
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
            rolling_loudness: RollingLoudness::new(),
            closed_detector: ClosedDetector::new(ClosedDetectionMode::UltraLowLatency),
            current_class: LipSyncClass::Rest,
            hold_time_seconds: 0.0,
            previous_time_step_seconds: 0.0,
            last_switch_confidence: 1.0,
            stream: AnalysisRing::default(),
            latest_frame: LipSyncFrame::default(),
            latest_debug_frame: LipSyncDebugFrame::default(),
            analysis_count: 0,
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
        self.process_debug(pcm_data).frame
    }

    pub fn process_debug(&mut self, pcm_data: &[f32]) -> LipSyncDebugFrame {
        let time_seconds = self.current_time_seconds;
        let debug_frame = self.process_at_time_debug(pcm_data, time_seconds);
        self.current_time_seconds += pcm_data.len() as f32 / self.options.sample_rate as f32;
        debug_frame
    }

    pub fn process_at_time(&mut self, pcm_data: &[f32], time_seconds: f32) -> LipSyncFrame {
        self.process_at_time_debug(pcm_data, time_seconds).frame
    }

    pub fn process_at_time_debug(
        &mut self,
        pcm_data: &[f32],
        time_seconds: f32,
    ) -> LipSyncDebugFrame {
        if pcm_data.is_empty() {
            return self.latest_debug_frame;
        }

        let ring_capacity_samples = self.ring_capacity_samples();
        self.stream.push(pcm_data, ring_capacity_samples);
        let window_samples = self.analysis_window_samples();
        let hop_samples = self.analysis_hop_samples();
        if !self.stream.has_enough_samples(window_samples) {
            self.latest_frame = LipSyncFrame::default();
            self.latest_debug_frame = LipSyncDebugFrame::default();
            return self.latest_debug_frame;
        }

        while let Some(window) = self.stream.next_due_window(window_samples, hop_samples) {
            self.latest_debug_frame = self.analyze_pcm_window(&window, time_seconds);
            self.latest_frame = self.latest_debug_frame.frame;
        }

        self.latest_debug_frame
    }

    fn analysis_window_samples(&self) -> usize {
        let window_ms = if self.options.singing_mode() {
            SINGING_WINDOW_MS
        } else {
            NORMAL_WINDOW_MS
        };
        ((self.options.sample_rate as f32 * window_ms / 1000.0).round() as usize).max(1)
    }

    fn analysis_hop_samples(&self) -> usize {
        ((self.options.sample_rate as f32 * ANALYSIS_HOP_MS / 1000.0).round() as usize).max(1)
    }

    fn ring_capacity_samples(&self) -> usize {
        (self.analysis_window_samples() * 4)
            .max(self.analysis_hop_samples() * 8)
            .max(FRAME_SIZE)
    }

    fn analyze_pcm_window(&mut self, pcm_data: &[f32], time_seconds: f32) -> LipSyncDebugFrame {
        let time_step_seconds = self.time_step_seconds(pcm_data);
        let profile = analyze_spectral_profile(pcm_data, self.options);
        let classifier_features: Option<&[f32]> = if self.options.gmm_enabled() {
            Some(&profile.normalized_bands)
        } else {
            None
        };
        let evidence = analyze_vowel_evidence_with_classifier_features(
            pcm_data,
            self.options,
            classifier_features,
        );
        self.update_loudness_trackers(&profile);
        self.analysis_count += 1;

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
            self.remember_rest_temporal(time_step_seconds);
            self.remember_frame(&frame);
            return LipSyncDebugFrame::from_parts(frame, evidence, &profile, 0.0);
        }

        let activity = self.activity_score(&profile);
        let rest_score = (1.0 - activity).clamp(0.0, 0.9);
        let mut fricative_score = fricative_score(&profile) * (1.0 - rest_score);
        if profile.compression_likelihood > 0.45 {
            fricative_score *= 1.0 - 0.25 * profile.compression_likelihood;
        }

        let openness = vowel_openness(evidence.scores);
        let closed_score = self.closed_detector.score(
            &profile,
            &evidence,
            openness,
            rest_score,
            fricative_score,
            activity,
        );
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

        self.apply_temporal_state(&mut posterior, time_step_seconds);

        let normalized_loudness = self
            .rolling_loudness
            .state_for(profile.rms)
            .normalized_level_01;
        let mut jaw_open = ((0.55 * openness + 0.45 * normalized_loudness)
            * (1.0 - posterior[LipSyncClass::Rest as usize])
            * (1.0 - posterior[LipSyncClass::Closed as usize] * 0.5))
            .clamp(0.0, 1.0);
        if self.options.robust_loudness_enabled() {
            let compression_jaw_floor = 0.10
                * profile.compression_likelihood
                * activity
                * (1.0 - posterior[LipSyncClass::Rest as usize])
                * (1.0 - posterior[LipSyncClass::Closed as usize]);
            jaw_open = jaw_open.max(compression_jaw_floor).clamp(0.0, 1.0);
        }
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
        LipSyncDebugFrame::from_parts(frame, evidence, &profile, activity)
    }

    fn time_step_seconds(&mut self, pcm_data: &[f32]) -> f32 {
        let dt = (pcm_data.len() as f32 / self.options.sample_rate as f32).clamp(0.005, 0.12);
        self.previous_time_step_seconds = dt;
        dt
    }

    fn remember_rest_temporal(&mut self, time_step_seconds: f32) {
        if self.current_class == LipSyncClass::Rest {
            self.hold_time_seconds += time_step_seconds;
        } else {
            self.current_class = LipSyncClass::Rest;
            self.hold_time_seconds = 0.0;
            self.last_switch_confidence = 1.0;
        }
    }

    fn apply_temporal_state(&mut self, posterior: &mut [f32; NUM_CLASSES], time_step_seconds: f32) {
        let candidate = best_class(posterior);
        let margin = class_switch_margin(posterior, self.current_class);
        let min_hold = minimum_hold_seconds(self.current_class, self.options.singing_mode());
        let threshold =
            switch_margin_threshold(self.current_class, candidate, self.options.singing_mode());

        self.hold_time_seconds += time_step_seconds;
        if candidate != self.current_class
            && self.hold_time_seconds >= min_hold
            && margin >= threshold
        {
            self.current_class = candidate;
            self.hold_time_seconds = 0.0;
            self.last_switch_confidence = posterior[candidate as usize];
        }

        apply_temporal_bias(
            posterior,
            self.current_class,
            candidate,
            self.options.singing_mode(),
        );
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
        self.rolling_loudness.update(profile.rms, activity > 0.2);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosedDetectionMode {
    UltraLowLatency,
    QualityLookahead,
}

#[derive(Debug, Clone)]
struct ClosedDetector {
    mode: ClosedDetectionMode,
    previous_rms: f32,
    previous_valley: f32,
}

impl ClosedDetector {
    fn new(mode: ClosedDetectionMode) -> Self {
        Self {
            mode,
            previous_rms: 0.0,
            previous_valley: 0.0,
        }
    }

    fn score(
        &mut self,
        profile: &SpectralProfile,
        evidence: &VowelEvidence,
        openness: f32,
        rest_score: f32,
        fricative_score: f32,
        activity: f32,
    ) -> f32 {
        if profile.rms <= SILENCE_RMS * 3.0 || rest_score > 0.72 {
            self.update(profile, 0.0);
            return 0.0;
        }

        let low_openness = (1.0 - openness).clamp(0.0, 1.0);
        let low_high_ratio = (1.0 - (profile.high_ratio / 0.35).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let compact_spectrum = (1.0 - profile.spectral_flatness).clamp(0.0, 1.0);
        let energy_valley = self.energy_valley(profile.rms);
        let following_onset = self.following_onset(profile.rms);
        let weak_confidence_hint = (1.0 - evidence.confidence).clamp(0.0, 1.0) * 0.08;
        let lookahead_bonus = if self.mode == ClosedDetectionMode::QualityLookahead {
            (self.previous_valley * following_onset * 0.35).clamp(0.0, 0.35)
        } else {
            0.0
        };

        let evidence_score = 0.30 * energy_valley
            + 0.22 * low_high_ratio
            + 0.18 * low_openness
            + 0.12 * compact_spectrum
            + 0.10 * following_onset
            + weak_confidence_hint
            + lookahead_bonus;
        let score = (evidence_score * activity).clamp(0.0, 0.65)
            * (1.0 - rest_score)
            * (1.0 - fricative_score * 0.85);
        self.update(profile, energy_valley);
        score.clamp(0.0, 0.65)
    }

    fn energy_valley(&self, rms: f32) -> f32 {
        if self.previous_rms <= SILENCE_RMS * 4.0 {
            return 0.0;
        }
        ((self.previous_rms * 0.65 - rms) / (self.previous_rms * 0.65 + EPSILON)).clamp(0.0, 1.0)
    }

    fn following_onset(&self, rms: f32) -> f32 {
        if self.previous_rms <= SILENCE_RMS * 2.0 {
            return 0.0;
        }
        ((rms - self.previous_rms * 1.35) / (self.previous_rms * 1.35 + EPSILON)).clamp(0.0, 1.0)
    }

    fn update(&mut self, profile: &SpectralProfile, energy_valley: f32) {
        self.previous_rms = profile.rms;
        self.previous_valley = energy_valley;
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
    analyze_vowel_evidence_with_classifier_features(pcm_data, options, None)
}

fn analyze_vowel_evidence_with_classifier_features(
    pcm_data: &[f32],
    options: LipSyncOptions,
    classifier_features: Option<&[f32]>,
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

    let mut scores = match VowelClassifierKind::from_options(options) {
        VowelClassifierKind::MultiPrototype => multi_prototype_scores(&profile.normalized_bands),
        VowelClassifierKind::DiagonalGmm => classifier_features
            .and_then(|features| gmm_vowel_scores(features).ok())
            .or_else(|| gmm_vowel_scores(&profile.normalized_bands).ok())
            .unwrap_or_else(|| multi_prototype_scores(&profile.normalized_bands)),
    };
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
        .max(0.08 * profile.compression_likelihood);
    let confidence_damping = compression_confidence_damping(profile.compression_likelihood);
    let confidence = (activity
        * (0.25 + separation * 2.8)
        * (1.0 - profile.spectral_flatness * 0.25)
        * confidence_damping)
        .clamp(0.0, 1.0);

    VowelEvidence {
        scores,
        confidence,
        f1_hz,
        f2_hz,
    }
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
    let target_start = if source.len() < FRAME_SIZE {
        (FRAME_SIZE - source.len()) / 2
    } else {
        0
    };
    for (index, sample) in source.iter().take(FRAME_SIZE).enumerate() {
        frame[target_start + index] = sample.clamp(-1.0, 1.0);
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

// Compression should preserve mouth-shape evidence. It smooths harsh spectral artifacts
// from clipping/limiting but does not push the feature toward a specific vowel.
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
            feature[index] * (1.0 - 0.16 * amount) + (left + right) * 0.08 * amount;
    }

    for index in 10..NUM_BANDS {
        let high_shelf = (index - 9) as f32 / (NUM_BANDS - 9) as f32;
        compensated[index] -= amount * high_shelf * 0.10;
    }

    normalize_feature(compensated)
}
fn gmm_vowel_scores(features: &[f32]) -> Result<[f32; NUM_VOWELS], gmm::GmmError> {
    let model = gmm::placeholder_vowel_gmm();
    model.posterior_checked::<NUM_VOWELS>(features)
}

fn multi_prototype_scores(feature: &[f32; NUM_BANDS]) -> [f32; NUM_VOWELS] {
    let mut logits = [0.0; NUM_VOWELS];
    for vowel_index in 0..NUM_VOWELS {
        let mut prototype_logits = [0.0; NUM_VOWEL_PROTOTYPES];
        for prototype_index in 0..NUM_VOWEL_PROTOTYPES {
            let prototype = normalized_prototype(VOWEL_PROTOTYPES[vowel_index][prototype_index]);
            prototype_logits[prototype_index] =
                cosine_similarity(feature, &prototype) * MULTI_PROTOTYPE_TEMPERATURE;
        }
        logits[vowel_index] = logsumexp(&prototype_logits);
    }
    softmax(logits, 1.0)
}

fn logsumexp(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_value = f32::NEG_INFINITY;
    for value in values {
        max_value = max_value.max(*value);
    }
    if !max_value.is_finite() {
        return max_value;
    }
    let mut sum = 0.0;
    for value in values {
        sum += (*value - max_value).exp();
    }
    max_value + (sum + EPSILON).ln()
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

fn compression_confidence_damping(compression_likelihood: f32) -> f32 {
    (1.0 - 0.22 * compression_likelihood.clamp(0.0, 1.0)).clamp(0.78, 1.0)
}

fn apply_compressed_voice_prior(scores: &mut [f32; NUM_VOWELS], compression_likelihood: f32) {
    let weight = (compression_likelihood.clamp(0.0, 1.0) * COMPRESSED_VOWEL_PRIOR_MAX_WEIGHT)
        .clamp(0.0, COMPRESSED_VOWEL_PRIOR_MAX_WEIGHT);
    if weight <= 0.01 {
        return;
    }

    // Keep this prior deliberately weak and broad. Compressed or shouted voices
    // should affect level/confidence handling more than vowel identity.
    let compressed_prior = [0.28, 0.13, 0.17, 0.20, 0.22];
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

fn best_class(posterior: &[f32; NUM_CLASSES]) -> LipSyncClass {
    let mut best_index = 0;
    let mut best_score = posterior[0];
    for (index, score) in posterior.iter().enumerate().skip(1) {
        if *score > best_score {
            best_index = index;
            best_score = *score;
        }
    }
    class_from_index(best_index)
}

fn class_switch_margin(posterior: &[f32; NUM_CLASSES], current: LipSyncClass) -> f32 {
    let candidate = best_class(posterior);
    posterior[candidate as usize] - posterior[current as usize]
}

fn class_from_index(index: usize) -> LipSyncClass {
    match index {
        1 => LipSyncClass::Closed,
        2 => LipSyncClass::A,
        3 => LipSyncClass::I,
        4 => LipSyncClass::U,
        5 => LipSyncClass::E,
        6 => LipSyncClass::O,
        7 => LipSyncClass::Fricative,
        8 => LipSyncClass::Other,
        _ => LipSyncClass::Rest,
    }
}

fn minimum_hold_seconds(class: LipSyncClass, singing_mode: bool) -> f32 {
    match class {
        LipSyncClass::Rest => 0.03,
        LipSyncClass::Closed => 0.04,
        LipSyncClass::Fricative => 0.05,
        LipSyncClass::A | LipSyncClass::I | LipSyncClass::U | LipSyncClass::E | LipSyncClass::O => {
            if singing_mode {
                0.09
            } else {
                0.055
            }
        }
        LipSyncClass::Other => 0.035,
    }
}

fn switch_margin_threshold(
    current: LipSyncClass,
    candidate: LipSyncClass,
    singing_mode: bool,
) -> f32 {
    if candidate == current {
        return 0.0;
    }
    if matches!(candidate, LipSyncClass::Closed | LipSyncClass::Fricative) {
        return 0.035;
    }
    if is_vowel_class(current) && is_vowel_class(candidate) {
        return if singing_mode { 0.16 } else { 0.07 };
    }
    if matches!(
        current,
        LipSyncClass::Rest | LipSyncClass::Closed | LipSyncClass::Fricative
    ) {
        return 0.045;
    }
    0.065
}

fn apply_temporal_bias(
    posterior: &mut [f32; NUM_CLASSES],
    current: LipSyncClass,
    candidate: LipSyncClass,
    singing_mode: bool,
) {
    let current_index = current as usize;
    let bias = if current == candidate {
        if matches!(current, LipSyncClass::Closed | LipSyncClass::Fricative) {
            0.04
        } else if is_vowel_class(current) && singing_mode {
            0.10
        } else {
            0.06
        }
    } else if is_vowel_class(current) && is_vowel_class(candidate) {
        if singing_mode { 0.18 } else { 0.08 }
    } else {
        0.05
    };
    posterior[current_index] += (1.0 - posterior[current_index]) * bias;
    normalize_distribution(posterior);
}

fn is_vowel_class(class: LipSyncClass) -> bool {
    matches!(
        class,
        LipSyncClass::A | LipSyncClass::I | LipSyncClass::U | LipSyncClass::E | LipSyncClass::O
    )
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
    fn multi_prototype_table_has_expected_shape() {
        assert_eq!(VOWEL_PROTOTYPES.len(), NUM_VOWELS);
        for prototypes in VOWEL_PROTOTYPES.iter() {
            assert_eq!(prototypes.len(), NUM_VOWEL_PROTOTYPES);
        }
    }

    #[test]
    fn multi_prototype_scores_are_normalized() {
        let feature = normalize_feature(VOWEL_PROTOTYPES[Vowel::A as usize][0]);
        let scores = multi_prototype_scores(&feature);
        assert_scores_normalized(scores);
        assert_eq!(best_vowel_from_scores(scores), Vowel::A);
    }

    #[test]
    fn multi_prototype_scoring_handles_zero_feature() {
        let scores = multi_prototype_scores(&[0.0; NUM_BANDS]);
        assert_scores_normalized(scores);
    }

    #[test]
    fn synthetic_vowel_like_signals_produce_non_uniform_scores() {
        let signals = [
            synthetic_vowel_like_signal(0.45, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]),
            synthetic_vowel_like_signal(0.45, &[(180.0, 0.25), (2200.0, 1.0), (3100.0, 0.6)]),
            synthetic_vowel_like_signal(0.45, &[(100.0, 0.7), (260.0, 1.0), (460.0, 0.8)]),
            synthetic_vowel_like_signal(0.45, &[(160.0, 0.25), (550.0, 0.8), (1900.0, 1.0)]),
            synthetic_vowel_like_signal(0.45, &[(120.0, 0.5), (430.0, 1.0), (900.0, 0.8)]),
        ];
        for signal in signals {
            let scores = analyze_vowel_evidence(&signal, SAMPLE_RATE).scores;
            assert_scores_normalized(scores);
            let (best, second) = best_two(scores);
            assert!(best > second + 0.01, "scores were too uniform: {scores:?}");
        }
    }

    #[test]
    fn gmm_classifier_can_be_selected_through_options() {
        let options = LipSyncOptions {
            sample_rate: SAMPLE_RATE,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS | LIPSYNC_FLAG_GMM,
            ..LipSyncOptions::default()
        };
        assert_eq!(
            VowelClassifierKind::from_options(options),
            VowelClassifierKind::DiagonalGmm
        );

        let pcm = synthetic_vowel_like_signal(0.45, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let scores = analyze_vowel_evidence_with_options(&pcm, options).scores;
        assert_scores_normalized(scores);
    }

    #[test]
    fn placeholder_gmm_rejects_mfcc_feature_vector_length() {
        let features = vec![0.0; crate::features::FEATURE_VECTOR_LEN];
        let err = gmm_vowel_scores(&features)
            .expect_err("31-dim MFCC feature vectors must not be accepted by 16-band GMM");
        assert_eq!(
            err,
            gmm::GmmError::FeatureDimensionMismatch {
                expected: gmm::PLACEHOLDER_GMM_FEATURES,
                actual: crate::features::FEATURE_VECTOR_LEN,
            }
        );
    }

    #[test]
    fn short_16khz_windows_are_center_padded() {
        let pcm = vec![0.1, -0.2, 0.3, -0.4];
        let frame = prepare_analysis_frame(&pcm, false);
        let target_start = (FRAME_SIZE - pcm.len()) / 2;
        assert_eq!(
            &frame[target_start..target_start + pcm.len()],
            pcm.as_slice()
        );
        assert!(frame[..target_start].iter().all(|sample| *sample == 0.0));
        assert!(
            frame[target_start + pcm.len()..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
    }

    #[test]
    fn small_eq_tilt_keeps_best_vowel_stable() {
        for vowel_index in 0..NUM_VOWELS {
            let base = normalize_feature(VOWEL_PROTOTYPES[vowel_index][0]);
            let mut tilted = base;
            for (band, value) in tilted.iter_mut().enumerate() {
                let position = band as f32 / (NUM_BANDS - 1) as f32;
                *value += (position - 0.5) * 0.12;
            }
            tilted = normalize_feature(tilted);
            let base_best = best_vowel_from_scores(multi_prototype_scores(&base));
            let tilted_best = best_vowel_from_scores(multi_prototype_scores(&tilted));
            assert_eq!(base_best, tilted_best);
        }
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
    fn clipped_or_compressed_voice_is_not_forced_to_a_by_prior() {
        let clean =
            synthetic_vowel_like_signal(0.35, &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)]);
        let clipped = clipped_signal(&clean, 5.0, 0.92);
        let clipped_scores = analyze_vowel_evidence(&clipped, SAMPLE_RATE).scores;
        assert_scores_normalized(clipped_scores);
        assert!(
            clipped_scores[Vowel::A as usize] < 0.45,
            "compressed prior over-forced A: {clipped_scores:?}"
        );
    }

    #[test]
    fn loud_clipped_i_like_signal_does_not_collapse_to_a() {
        let clean =
            synthetic_vowel_like_signal(0.45, &[(180.0, 0.25), (2200.0, 1.0), (3100.0, 0.6)]);
        let clipped = clipped_signal(&clean, 6.0, 0.96);
        let scores = analyze_vowel_evidence(&clipped, SAMPLE_RATE).scores;
        assert_scores_normalized(scores);
        assert_ne!(
            best_vowel_from_scores(scores),
            Vowel::A,
            "I-like clipped voice collapsed to A: {scores:?}"
        );
    }

    #[test]
    fn loud_clipped_u_like_signal_does_not_collapse_to_a() {
        let clean = synthetic_vowel_like_signal(0.45, &[(100.0, 0.7), (260.0, 1.0), (460.0, 0.8)]);
        let clipped = clipped_signal(&clean, 6.0, 0.96);
        let scores = analyze_vowel_evidence(&clipped, SAMPLE_RATE).scores;
        assert_scores_normalized(scores);
        assert_ne!(
            best_vowel_from_scores(scores),
            Vowel::A,
            "U-like clipped voice collapsed to A: {scores:?}"
        );
    }

    #[test]
    fn compressed_voice_prior_keeps_scores_normalized_without_dominating_identity() {
        let mut scores = [0.03, 0.64, 0.24, 0.05, 0.04];
        apply_compressed_voice_prior(&mut scores, 1.0);
        assert_scores_normalized(scores);
        assert_eq!(best_vowel_from_scores(scores), Vowel::I);
        assert!(
            scores[Vowel::A as usize] < 0.12,
            "weak prior over-weighted A: {scores:?}"
        );
    }

    #[test]
    fn compression_confidence_damping_is_controlled() {
        let clean = compression_confidence_damping(0.0);
        let compressed = compression_confidence_damping(1.0);
        assert_eq!(clean, 1.0);
        assert!(compressed < clean);
        assert!(compressed >= 0.78);
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
    fn analyzer_returns_rest_and_closed_classes_for_zero_pcm() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let frame = analyzer.process(&vec![0.0; FRAME_SIZE]);
        assert!(frame.posterior[LipSyncClass::Rest as usize] > 0.9);
        assert!(frame.posterior[LipSyncClass::Closed as usize] <= 0.05);
        assert!(frame.jaw_open <= 0.01);
        assert_scores_normalized(frame.posterior);
    }

    #[test]
    fn silence_is_rest_not_closed() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let frame = analyzer.process(&vec![0.0; FRAME_SIZE]);
        assert!(frame.posterior[LipSyncClass::Rest as usize] > 0.9);
        assert!(frame.posterior[LipSyncClass::Closed as usize] < 0.05);
    }

    #[test]
    fn high_frequency_noise_prefers_fricative_over_vowels() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let noise = synthetic_high_frequency_noise(0.35);
        let frame = analyzer.process(&analysis_ready_signal(noise));
        let best_vowel = frame.posterior[LipSyncClass::A as usize..=LipSyncClass::O as usize]
            .iter()
            .copied()
            .fold(0.0, f32::max);
        assert!(frame.posterior[LipSyncClass::Fricative as usize] > best_vowel);
        assert!(
            frame.posterior[LipSyncClass::Fricative as usize]
                > frame.posterior[LipSyncClass::Closed as usize]
        );
    }

    #[test]
    fn energy_valley_followed_by_onset_boosts_closed_in_quality_mode() {
        let evidence = VowelEvidence {
            scores: [0.16, 0.24, 0.30, 0.16, 0.14],
            confidence: 0.25,
            f1_hz: 0.0,
            f2_hz: 0.0,
        };
        let first = synthetic_closed_profile(0.16, 0.18, 0.08, 0.08);
        let valley = synthetic_closed_profile(0.025, 0.03, 0.06, 0.05);
        let onset = synthetic_closed_profile(0.15, 0.20, 0.08, 0.10);

        let mut ultra = ClosedDetector::new(ClosedDetectionMode::UltraLowLatency);
        ultra.score(&first, &evidence, 0.25, 0.05, 0.02, 1.0);
        ultra.score(&valley, &evidence, 0.20, 0.05, 0.02, 1.0);
        let ultra_onset = ultra.score(&onset, &evidence, 0.20, 0.05, 0.02, 1.0);

        let mut quality = ClosedDetector::new(ClosedDetectionMode::QualityLookahead);
        quality.score(&first, &evidence, 0.25, 0.05, 0.02, 1.0);
        quality.score(&valley, &evidence, 0.20, 0.05, 0.02, 1.0);
        let quality_onset = quality.score(&onset, &evidence, 0.20, 0.05, 0.02, 1.0);

        assert!(
            quality_onset > ultra_onset,
            "quality={quality_onset} ultra={ultra_onset}"
        );
    }

    #[test]
    fn sustained_low_openness_vowel_does_not_become_closed() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let u_like = synthetic_vowel_like_signal(0.45, &[(100.0, 0.7), (260.0, 1.0), (460.0, 0.8)]);
        let frame = analyzer.process(&analysis_ready_signal(u_like));
        let best_vowel = frame.posterior[LipSyncClass::A as usize..=LipSyncClass::O as usize]
            .iter()
            .copied()
            .fold(0.0, f32::max);
        assert!(best_vowel > frame.posterior[LipSyncClass::Closed as usize]);
    }

    #[test]
    fn voiced_signal_prefers_a_vowel_class() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let voiced =
            synthetic_vowel_like_signal(0.5, &[(120.0, 0.4), (720.0, 1.0), (1150.0, 0.65)]);
        let frame = analyzer.process(&analysis_ready_signal(voiced));
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
        normal.process(&analysis_ready_signal(a_like.clone()));
        singing.process(&analysis_ready_signal(a_like));
        let normal_i = normal
            .process(&analysis_ready_signal(i_like.clone()))
            .posterior[LipSyncClass::I as usize];
        let singing_i =
            singing.process(&analysis_ready_signal(i_like)).posterior[LipSyncClass::I as usize];
        assert!(singing_i < normal_i);
    }

    #[test]
    fn rapid_vowel_switches_switch_less_in_singing_mode() {
        let mut normal = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let mut singing = LipSyncAnalyzer::new(SAMPLE_RATE, true);
        let sequence = [
            LipSyncClass::A,
            LipSyncClass::I,
            LipSyncClass::A,
            LipSyncClass::I,
            LipSyncClass::A,
            LipSyncClass::I,
        ];
        let normal_switches = count_temporal_switches(&mut normal, &sequence, 0.06);
        let singing_switches = count_temporal_switches(&mut singing, &sequence, 0.06);
        assert!(singing_switches < normal_switches);
    }

    #[test]
    fn normal_mode_switches_after_minimum_hold() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let mut a = posterior_for(LipSyncClass::A, 0.82);
        analyzer.apply_temporal_state(&mut a, 0.06);
        assert_eq!(analyzer.current_class, LipSyncClass::A);

        let mut i = posterior_for(LipSyncClass::I, 0.82);
        analyzer.apply_temporal_state(&mut i, 0.06);
        assert_eq!(analyzer.current_class, LipSyncClass::I);
    }

    #[test]
    fn closed_and_fricative_can_attack_quickly() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let mut a = posterior_for(LipSyncClass::A, 0.82);
        analyzer.apply_temporal_state(&mut a, 0.06);

        let mut closed = posterior_pair(LipSyncClass::A, 0.20, LipSyncClass::Closed, 0.72);
        analyzer.apply_temporal_state(&mut closed, 0.06);
        assert_eq!(analyzer.current_class, LipSyncClass::Closed);

        let mut fricative =
            posterior_pair(LipSyncClass::Closed, 0.20, LipSyncClass::Fricative, 0.72);
        analyzer.apply_temporal_state(&mut fricative, 0.05);
        assert_eq!(analyzer.current_class, LipSyncClass::Fricative);
    }

    #[test]
    fn low_confidence_candidate_does_not_replace_current_class() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let mut a = posterior_for(LipSyncClass::A, 0.82);
        analyzer.apply_temporal_state(&mut a, 0.06);
        let mut weak_i = posterior_pair(LipSyncClass::A, 0.40, LipSyncClass::I, 0.43);
        analyzer.apply_temporal_state(&mut weak_i, 0.12);
        assert_eq!(analyzer.current_class, LipSyncClass::A);
    }

    #[test]
    fn temporal_processing_keeps_posterior_normalized_and_jaw_clamped() {
        let mut analyzer = LipSyncAnalyzer::new(SAMPLE_RATE, false);
        let voiced =
            synthetic_vowel_like_signal(0.5, &[(120.0, 0.4), (720.0, 1.0), (1150.0, 0.65)]);
        let frame = analyzer.process(&analysis_ready_signal(voiced));
        assert_scores_normalized(frame.posterior);
        assert!(frame.jaw_open.is_finite());
        assert!((0.0..=1.0).contains(&frame.jaw_open));
    }

    #[test]
    fn processing_1024_once_is_similar_to_128_sample_chunks() {
        let sample_rate = 16_000;
        let signal = synthetic_vowel_like_signal_at(
            sample_rate,
            1024,
            0.45,
            &[(120.0, 0.35), (720.0, 1.0), (1150.0, 0.75)],
        );
        let mut whole = LipSyncAnalyzer::new(sample_rate, false);
        let whole_frame = whole.process(&signal);

        let mut chunked = LipSyncAnalyzer::new(sample_rate, false);
        let mut chunked_frame = LipSyncFrame::default();
        for chunk in signal.chunks(128) {
            chunked_frame = chunked.process(chunk);
        }

        let drift = whole_frame
            .posterior
            .iter()
            .zip(chunked_frame.posterior.iter())
            .map(|(whole, chunked)| (whole - chunked).abs())
            .sum::<f32>();
        assert!(drift < 0.35, "chunked drift was {drift}");
    }

    #[test]
    fn one_20ms_chunk_does_not_reanalyze_identical_latest_window_twice() {
        let sample_rate = 16_000;
        let mut analyzer = LipSyncAnalyzer::new(sample_rate, false);
        let chunk = synthetic_vowel_like_signal_at(sample_rate, 320, 0.35, &[(220.0, 1.0)]);

        analyzer.process(&chunk);
        assert_eq!(analyzer.analysis_count, 0);

        analyzer.process(&chunk);
        assert_eq!(analyzer.analysis_count, 2);
        assert_eq!(analyzer.stream.previous_window_end_sample, 400);
        assert_eq!(analyzer.stream.last_window_end_sample, 560);
    }

    #[test]
    fn chunks_smaller_than_hop_do_not_panic() {
        let mut analyzer = LipSyncAnalyzer::new(16_000, false);
        let chunk = synthetic_vowel_like_signal_at(16_000, 64, 0.35, &[(220.0, 1.0)]);
        for _ in 0..20 {
            let frame = analyzer.process(&chunk);
            assert_scores_normalized(frame.posterior);
        }
    }

    #[test]
    fn sample_rates_produce_reasonable_window_sizes() {
        let normal_16k = LipSyncAnalyzer::new(16_000, false);
        let normal_48k = LipSyncAnalyzer::new(48_000, false);
        assert_eq!(normal_16k.analysis_window_samples(), 400);
        assert_eq!(normal_48k.analysis_window_samples(), 1200);
        assert_eq!(normal_16k.analysis_hop_samples(), 160);
        assert_eq!(normal_48k.analysis_hop_samples(), 480);
    }

    #[test]
    fn silence_warmup_returns_rest() {
        let mut analyzer = LipSyncAnalyzer::new(48_000, false);
        let frame = analyzer.process(&vec![0.0; 128]);
        assert!(frame.posterior[LipSyncClass::Rest as usize] > 0.9);
    }

    #[test]
    fn ring_buffer_does_not_grow_unbounded() {
        let mut analyzer = LipSyncAnalyzer::new(16_000, false);
        let capacity = analyzer.ring_capacity_samples();
        let chunk = synthetic_vowel_like_signal_at(16_000, 320, 0.2, &[(180.0, 1.0)]);
        for _ in 0..100 {
            analyzer.process(&chunk);
        }
        assert!(analyzer.stream.len() <= capacity);
    }

    #[test]
    fn singing_mode_uses_longer_window_than_normal_mode() {
        let normal = LipSyncAnalyzer::new(16_000, false);
        let singing = LipSyncAnalyzer::new(16_000, true);
        assert!(singing.analysis_window_samples() > normal.analysis_window_samples());
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
        let frame = analyzer.process_at_time(&analysis_ready_signal(pcm), 0.2);
        assert!(
            frame.posterior[LipSyncClass::O as usize] > frame.posterior[LipSyncClass::A as usize]
        );
    }

    fn count_temporal_switches(
        analyzer: &mut LipSyncAnalyzer,
        sequence: &[LipSyncClass],
        dt: f32,
    ) -> usize {
        let mut switches = 0;
        let mut previous = analyzer.current_class;
        for class in sequence {
            let mut posterior = posterior_for(*class, 0.82);
            analyzer.apply_temporal_state(&mut posterior, dt);
            if analyzer.current_class != previous {
                switches += 1;
                previous = analyzer.current_class;
            }
        }
        switches
    }

    fn posterior_for(class: LipSyncClass, score: f32) -> [f32; NUM_CLASSES] {
        let mut posterior = [0.0; NUM_CLASSES];
        posterior[LipSyncClass::Rest as usize] = 0.03;
        posterior[LipSyncClass::Other as usize] = (1.0 - score - 0.03).max(0.0);
        posterior[class as usize] = score;
        normalize_distribution(&mut posterior);
        posterior
    }

    fn posterior_pair(
        current: LipSyncClass,
        current_score: f32,
        candidate: LipSyncClass,
        candidate_score: f32,
    ) -> [f32; NUM_CLASSES] {
        let mut posterior = [0.0; NUM_CLASSES];
        posterior[current as usize] = current_score;
        posterior[candidate as usize] = candidate_score;
        posterior[LipSyncClass::Other as usize] = (1.0 - current_score - candidate_score).max(0.0);
        normalize_distribution(&mut posterior);
        posterior
    }

    fn analysis_ready_signal(mut signal: Vec<f32>) -> Vec<f32> {
        let original = signal.clone();
        signal.extend_from_slice(&original);
        signal
    }

    fn synthetic_closed_profile(
        rms: f32,
        peak: f32,
        high_ratio: f32,
        spectral_flatness: f32,
    ) -> SpectralProfile {
        SpectralProfile {
            rms,
            peak,
            normalized_bands: [0.0; NUM_BANDS],
            high_ratio,
            zero_crossing_rate: 0.02,
            spectral_flatness,
            compression_likelihood: 0.0,
        }
    }

    fn synthetic_vowel_like_signal_at(
        sample_rate: u32,
        len: usize,
        amplitude: f32,
        partials: &[(f32, f32)],
    ) -> Vec<f32> {
        (0..len)
            .map(|sample_index| {
                let time = sample_index as f32 / sample_rate as f32;
                let sum = partials
                    .iter()
                    .map(|(hz, weight)| (2.0 * std::f32::consts::PI * hz * time).sin() * weight)
                    .sum::<f32>();
                (sum * amplitude / partials.len() as f32).clamp(-1.0, 1.0)
            })
            .collect()
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

    fn clipped_signal(samples: &[f32], gain: f32, limit: f32) -> Vec<f32> {
        samples
            .iter()
            .map(|sample| (*sample * gain).clamp(-limit, limit))
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

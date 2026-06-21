const EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone)]
pub struct RollingCmvn {
    mean: Vec<f32>,
    variance: Vec<f32>,
    initialized: bool,
    update_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct CmvnUpdateGate {
    pub voiced_confidence: f32,
    pub rest_score: f32,
    pub fricative_score: f32,
    pub compression_likelihood: f32,
}

impl RollingCmvn {
    pub fn new(update_rate: f32) -> Self {
        Self {
            mean: Vec::new(),
            variance: Vec::new(),
            initialized: false,
            update_rate: update_rate.clamp(0.001, 1.0),
        }
    }

    pub fn normalize(&self, features: &[f32]) -> Vec<f32> {
        if features.is_empty() {
            return Vec::new();
        }
        if !self.initialized
            || self.mean.len() != features.len()
            || self.variance.len() != features.len()
        {
            return sanitize(features);
        }
        features
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let std = self.variance[index].max(EPSILON).sqrt();
                ((finite_or_zero(*value) - self.mean[index]) / std).clamp(-6.0, 6.0)
            })
            .collect()
    }

    pub fn update(&mut self, features: &[f32], update_weight: f32) {
        if features.is_empty() || !features.iter().all(|value| value.is_finite()) {
            return;
        }
        if !self.initialized || self.mean.len() != features.len() {
            self.mean = sanitize(features);
            self.variance = vec![1.0; features.len()];
            self.initialized = true;
            return;
        }

        let weight = (update_weight * self.update_rate).clamp(0.0, 1.0);
        if weight <= 0.0 {
            return;
        }
        for (index, feature) in features.iter().enumerate() {
            let value = finite_or_zero(*feature);
            let delta = value - self.mean[index];
            self.mean[index] += delta * weight;
            let variance_delta = delta * delta - self.variance[index];
            self.variance[index] = (self.variance[index] + variance_delta * weight).max(EPSILON);
        }
    }

    pub fn update_if_reliable(&mut self, features: &[f32], gate: CmvnUpdateGate) {
        if !gate.is_reliable() {
            return;
        }
        self.update(features, gate.update_weight());
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for RollingCmvn {
    fn default() -> Self {
        Self::new(0.04)
    }
}

impl CmvnUpdateGate {
    pub fn is_reliable(self) -> bool {
        self.voiced_confidence >= 0.45
            && self.rest_score <= 0.45
            && self.fricative_score <= 0.35
            && self.compression_likelihood <= 0.65
    }

    pub fn update_weight(self) -> f32 {
        let voiced = ((self.voiced_confidence - 0.45) / 0.45).clamp(0.0, 1.0);
        let rest = (1.0 - self.rest_score).clamp(0.0, 1.0);
        let fricative = (1.0 - self.fricative_score).clamp(0.0, 1.0);
        let compression = (1.0 - self.compression_likelihood).clamp(0.0, 1.0);
        (voiced * rest * fricative * compression).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct RollingLoudness {
    noise_floor: f32,
    speech_high: f32,
    initialized: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct LoudnessState {
    pub normalized_level_01: f32,
    pub noise_floor: f32,
    pub speech_high: f32,
}

impl RollingLoudness {
    pub fn new() -> Self {
        Self {
            noise_floor: 0.0005,
            speech_high: 0.08,
            initialized: false,
        }
    }

    pub fn update(&mut self, rms: f32, active: bool) -> LoudnessState {
        let rms = finite_or_zero(rms).max(0.0);
        if !self.initialized {
            self.noise_floor = rms.min(0.01).max(0.0005);
            self.speech_high = rms.max(0.04);
            self.initialized = true;
        }

        if active {
            let attack = if rms > self.speech_high { 0.08 } else { 0.015 };
            self.speech_high =
                self.speech_high * (1.0 - attack) + rms.max(self.noise_floor) * attack;
        } else {
            self.noise_floor = self.noise_floor * 0.97 + rms.min(self.speech_high) * 0.03;
        }
        self.noise_floor = self.noise_floor.clamp(0.0001, 0.05);
        self.speech_high = self
            .speech_high
            .max(self.noise_floor * 4.0)
            .clamp(0.002, 1.0);
        self.state_for(rms)
    }

    pub fn state_for(&self, rms: f32) -> LoudnessState {
        let span = (self.speech_high - self.noise_floor).max(EPSILON);
        LoudnessState {
            normalized_level_01: ((rms - self.noise_floor) / span).clamp(0.0, 1.0),
            noise_floor: self.noise_floor,
            speech_high: self.speech_high,
        }
    }
}

impl Default for RollingLoudness {
    fn default() -> Self {
        Self::new()
    }
}

fn sanitize(features: &[f32]) -> Vec<f32> {
    features
        .iter()
        .map(|value| finite_or_zero(*value))
        .collect()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmvn_output_is_finite() {
        let mut cmvn = RollingCmvn::new(0.1);
        cmvn.update(&[1.0, 2.0, 3.0], 1.0);
        let normalized = cmvn.normalize(&[1.5, 2.5, 3.5]);
        assert_eq!(normalized.len(), 3);
        assert!(normalized.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn cmvn_adapts_over_repeated_voiced_frames() {
        let mut cmvn = RollingCmvn::new(0.5);
        let gate = CmvnUpdateGate {
            voiced_confidence: 0.9,
            rest_score: 0.05,
            fricative_score: 0.05,
            compression_likelihood: 0.0,
        };
        for _ in 0..8 {
            cmvn.update_if_reliable(&[2.0, 4.0], gate);
        }
        assert!(cmvn.is_initialized());
        let normalized = cmvn.normalize(&[2.0, 4.0]);
        assert!(normalized.iter().all(|value| value.abs() < 0.5));
    }

    #[test]
    fn cmvn_does_not_update_on_silence() {
        let mut cmvn = RollingCmvn::new(0.5);
        let gate = CmvnUpdateGate {
            voiced_confidence: 0.1,
            rest_score: 0.95,
            fricative_score: 0.0,
            compression_likelihood: 0.0,
        };
        cmvn.update_if_reliable(&[10.0, 20.0], gate);
        assert!(!cmvn.is_initialized());
    }

    #[test]
    fn loudness_normalization_keeps_quiet_and_loud_comparable() {
        let mut loudness = RollingLoudness::new();
        for _ in 0..20 {
            loudness.update(0.003, false);
        }
        let quiet = loudness.update(0.02, true).normalized_level_01;
        for _ in 0..20 {
            loudness.update(0.2, true);
        }
        let loud = loudness.update(0.22, true).normalized_level_01;
        assert!((0.0..=1.0).contains(&quiet));
        assert!((0.0..=1.0).contains(&loud));
        assert!(loud >= quiet * 0.5);
    }

    #[test]
    fn no_panic_on_empty_or_mismatched_feature_lengths() {
        let mut cmvn = RollingCmvn::new(0.1);
        cmvn.update(&[], 1.0);
        assert!(cmvn.normalize(&[]).is_empty());
        cmvn.update(&[1.0, 2.0, 3.0], 1.0);
        let normalized = cmvn.normalize(&[1.0, 2.0]);
        assert_eq!(normalized.len(), 2);
    }
}

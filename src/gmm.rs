const LOG_2_PI: f32 = 1.837_877;
const VARIANCE_FLOOR: f32 = 1.0e-4;
const EPSILON: f32 = 1.0e-8;

pub struct DiagonalGmm {
    pub num_classes: usize,
    pub num_mixtures: usize,
    pub num_features: usize,
    pub means: &'static [f32],
    pub inv_vars: &'static [f32],
    pub log_weights: &'static [f32],
    pub class_priors: &'static [f32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmmError {
    ClassCountMismatch { expected: usize, actual: usize },
    FeatureDimensionMismatch { expected: usize, actual: usize },
    InvalidModel,
}

impl DiagonalGmm {
    pub fn validate<const N: usize>(&self, features: &[f32]) -> Result<(), GmmError> {
        if self.num_classes != N {
            return Err(GmmError::ClassCountMismatch {
                expected: self.num_classes,
                actual: N,
            });
        }
        if features.len() != self.num_features {
            return Err(GmmError::FeatureDimensionMismatch {
                expected: self.num_features,
                actual: features.len(),
            });
        }
        if self.num_mixtures == 0 || self.num_features == 0 {
            return Err(GmmError::InvalidModel);
        }

        let expected_params = self.num_classes * self.num_mixtures * self.num_features;
        let expected_mixtures = self.num_classes * self.num_mixtures;
        if self.means.len() < expected_params
            || self.inv_vars.len() < expected_params
            || self.log_weights.len() < expected_mixtures
            || self.class_priors.len() < self.num_classes
        {
            return Err(GmmError::InvalidModel);
        }

        Ok(())
    }

    pub fn score_class(&self, class_index: usize, features: &[f32]) -> f32 {
        if class_index >= self.num_classes || features.len() != self.num_features {
            return f32::NEG_INFINITY;
        }

        let expected_params = self.num_classes * self.num_mixtures * self.num_features;
        let expected_mixtures = self.num_classes * self.num_mixtures;
        if self.means.len() < expected_params
            || self.inv_vars.len() < expected_params
            || self.log_weights.len() < expected_mixtures
            || self.class_priors.len() < self.num_classes
        {
            return f32::NEG_INFINITY;
        }

        let mut mixture_scores = vec![f32::NEG_INFINITY; self.num_mixtures];
        for mixture_index in 0..self.num_mixtures {
            let mixture_offset =
                (class_index * self.num_mixtures + mixture_index) * self.num_features;
            let mut score = self.log_weights[class_index * self.num_mixtures + mixture_index];
            for feature_index in 0..self.num_features {
                let feature = finite_or_zero(features[feature_index]);
                let mean = finite_or_zero(self.means[mixture_offset + feature_index]);
                let inv_var = finite_or_default(self.inv_vars[mixture_offset + feature_index], 1.0)
                    .max(VARIANCE_FLOOR);
                let diff = feature - mean;
                score += -0.5 * (diff * diff * inv_var - inv_var.ln() + LOG_2_PI);
            }
            mixture_scores[mixture_index] = score;
        }
        logsumexp(&mixture_scores) + finite_or_zero(self.class_priors[class_index])
    }

    pub fn posterior_checked<const N: usize>(
        &self,
        features: &[f32],
    ) -> Result<[f32; N], GmmError> {
        self.validate::<N>(features)?;

        let mut logits = [f32::NEG_INFINITY; N];
        for (class_index, logit) in logits.iter_mut().enumerate() {
            *logit = self.score_class(class_index, features);
        }
        Ok(softmax_logits(logits))
    }

    pub fn posterior<const N: usize>(&self, features: &[f32]) -> [f32; N] {
        self.posterior_checked::<N>(features)
            .unwrap_or_else(|_| uniform())
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn finite_or_default(value: f32, default: f32) -> f32 {
    if value.is_finite() { value } else { default }
}

fn uniform<const N: usize>() -> [f32; N] {
    if N == 0 {
        return [0.0; N];
    }
    [1.0 / N as f32; N]
}

fn softmax_logits<const N: usize>(logits: [f32; N]) -> [f32; N] {
    let max_value = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_value.is_finite() {
        return uniform();
    }

    let mut output = [0.0; N];
    let mut sum = 0.0;
    for index in 0..N {
        output[index] = (logits[index] - max_value).exp();
        sum += output[index];
    }
    if sum <= EPSILON || !sum.is_finite() {
        return uniform();
    }
    for value in &mut output {
        *value /= sum;
    }
    output
}

pub fn logsumexp(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_value.is_finite() {
        return max_value;
    }
    let sum = values
        .iter()
        .map(|value| (*value - max_value).exp())
        .sum::<f32>();
    max_value + (sum + EPSILON).ln()
}

pub const PLACEHOLDER_GMM_CLASSES: usize = 5;
pub const PLACEHOLDER_GMM_MIXTURES: usize = 2;
pub const PLACEHOLDER_GMM_FEATURES: usize = 16;

// Seeded from the same hand-written spectral prototype family used by the
// multi-prototype classifier. It is infrastructure, not a trained accuracy claim.
pub static PLACEHOLDER_GMM_MEANS: [f32; PLACEHOLDER_GMM_CLASSES
    * PLACEHOLDER_GMM_MIXTURES
    * PLACEHOLDER_GMM_FEATURES] = [
    0.1, 0.2, 0.45, 0.8, 1.15, 1.35, 1.05, 0.85, 0.55, 0.25, 0.0, -0.25, -0.45, -0.65, -0.85, -1.0,
    0.26, 0.34, 0.57, 0.9, 1.21, 1.39, 1.07, 0.85, 0.53, 0.21, -0.06, -0.33, -0.55, -0.77, -0.99,
    -1.16, -0.95, -0.8, -0.65, -0.5, -0.35, -0.15, 0.05, 0.35, 0.75, 1.1, 1.35, 1.5, 1.15, 0.75,
    0.25, -0.2, -1.11, -0.94, -0.77, -0.6, -0.41, -0.19, 0.03, 0.35, 0.77, 1.14, 1.41, 1.58, 1.25,
    0.87, 0.39, -0.04, 0.65, 1.05, 1.25, 1.05, 0.8, 0.45, 0.15, -0.1, -0.3, -0.5, -0.75, -0.95,
    -1.1, -1.2, -1.25, -1.3, 0.81, 1.19, 1.37, 1.15, 0.86, 0.49, 0.17, -0.1, -0.32, -0.54, -0.81,
    -1.03, -1.2, -1.32, -1.39, -1.46, -0.35, -0.15, 0.05, 0.3, 0.55, 0.75, 0.85, 0.95, 1.1, 1.15,
    0.95, 0.65, 0.25, -0.15, -0.5, -0.8, -0.51, -0.29, -0.07, 0.2, 0.49, 0.71, 0.83, 0.95, 1.12,
    1.19, 1.01, 0.73, 0.35, -0.03, -0.36, -0.64, 0.35, 0.75, 1.0, 0.95, 0.85, 0.7, 0.5, 0.3, 0.05,
    -0.15, -0.4, -0.65, -0.85, -1.0, -1.1, -1.2, 0.51, 0.89, 1.12, 1.05, 0.91, 0.74, 0.52, 0.3,
    0.03, -0.19, -0.46, -0.73, -0.95, -1.12, -1.24, -1.36,
];

pub static PLACEHOLDER_GMM_INV_VARS: [f32; PLACEHOLDER_GMM_CLASSES
    * PLACEHOLDER_GMM_MIXTURES
    * PLACEHOLDER_GMM_FEATURES] =
    [4.0; PLACEHOLDER_GMM_CLASSES * PLACEHOLDER_GMM_MIXTURES * PLACEHOLDER_GMM_FEATURES];
pub static PLACEHOLDER_GMM_LOG_WEIGHTS: [f32; PLACEHOLDER_GMM_CLASSES * PLACEHOLDER_GMM_MIXTURES] =
    [-0.693_147; PLACEHOLDER_GMM_CLASSES * PLACEHOLDER_GMM_MIXTURES];
pub static PLACEHOLDER_GMM_CLASS_PRIORS: [f32; PLACEHOLDER_GMM_CLASSES] =
    [0.0; PLACEHOLDER_GMM_CLASSES];

pub fn placeholder_vowel_gmm() -> DiagonalGmm {
    DiagonalGmm {
        num_classes: PLACEHOLDER_GMM_CLASSES,
        num_mixtures: PLACEHOLDER_GMM_MIXTURES,
        num_features: PLACEHOLDER_GMM_FEATURES,
        means: &PLACEHOLDER_GMM_MEANS,
        inv_vars: &PLACEHOLDER_GMM_INV_VARS,
        log_weights: &PLACEHOLDER_GMM_LOG_WEIGHTS,
        class_priors: &PLACEHOLDER_GMM_CLASS_PRIORS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmm_posterior_normalizes() {
        let gmm = placeholder_vowel_gmm();
        let posterior = gmm.posterior::<PLACEHOLDER_GMM_CLASSES>(&[0.0; PLACEHOLDER_GMM_FEATURES]);
        assert_normalized(posterior);
    }

    #[test]
    fn gmm_handles_invalid_features_safely() {
        let gmm = placeholder_vowel_gmm();
        let mut features = [0.0; PLACEHOLDER_GMM_FEATURES];
        features[3] = f32::NAN;
        features[7] = f32::INFINITY;
        let posterior = gmm.posterior::<PLACEHOLDER_GMM_CLASSES>(&features);
        assert_normalized(posterior);
    }

    #[test]
    fn gmm_rejects_feature_dimension_mismatch() {
        let gmm = placeholder_vowel_gmm();
        let err = gmm
            .posterior_checked::<PLACEHOLDER_GMM_CLASSES>(&[0.0; PLACEHOLDER_GMM_FEATURES + 1])
            .expect_err("dimension mismatch should not be silently accepted");
        assert_eq!(
            err,
            GmmError::FeatureDimensionMismatch {
                expected: PLACEHOLDER_GMM_FEATURES,
                actual: PLACEHOLDER_GMM_FEATURES + 1,
            }
        );
    }

    #[test]
    fn gmm_does_not_panic_on_tiny_variance() {
        static MEANS: [f32; 2] = [0.0, 1.0];
        static INV_VARS: [f32; 2] = [1.0e12, 0.0];
        static LOG_WEIGHTS: [f32; 2] = [0.0, 0.0];
        static PRIORS: [f32; 2] = [0.0, 0.0];
        let gmm = DiagonalGmm {
            num_classes: 2,
            num_mixtures: 1,
            num_features: 1,
            means: &MEANS,
            inv_vars: &INV_VARS,
            log_weights: &LOG_WEIGHTS,
            class_priors: &PRIORS,
        };
        let posterior = gmm.posterior::<2>(&[0.5]);
        assert_normalized(posterior);
    }

    fn assert_normalized<const N: usize>(scores: [f32; N]) {
        let sum = scores.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 0.001, "score sum was {sum}");
        for score in scores {
            assert!(score.is_finite());
            assert!((0.0..=1.0).contains(&score));
        }
    }
}

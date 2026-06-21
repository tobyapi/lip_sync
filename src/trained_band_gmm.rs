// Generated model slot for the trained 16-band vowel GMM.
// Empty arrays intentionally fall back to gmm::placeholder_vowel_gmm(), which is
// infrastructure only and not an accuracy model.

use crate::gmm::{DiagonalGmm, placeholder_vowel_gmm};

pub const TRAINED_BAND_GMM_CLASSES: [&str; 0] = [];
pub const TRAINED_BAND_GMM_MIXTURES: usize = 0;
pub const TRAINED_BAND_GMM_FEATURES: usize = 0;

pub const TRAINED_BAND_GMM_MEANS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_INV_VARS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_LOG_WEIGHTS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_CLASS_PRIORS: [f32; 0] = [];

pub fn trained_band_vowel_gmm() -> DiagonalGmm {
    let num_classes = TRAINED_BAND_GMM_CLASSES.len();
    let expected_params = num_classes * TRAINED_BAND_GMM_MIXTURES * TRAINED_BAND_GMM_FEATURES;
    let expected_mixtures = num_classes * TRAINED_BAND_GMM_MIXTURES;

    if num_classes == 0
        || TRAINED_BAND_GMM_MIXTURES == 0
        || TRAINED_BAND_GMM_FEATURES == 0
        || TRAINED_BAND_GMM_MEANS.len() != expected_params
        || TRAINED_BAND_GMM_INV_VARS.len() != expected_params
        || TRAINED_BAND_GMM_LOG_WEIGHTS.len() != expected_mixtures
        || TRAINED_BAND_GMM_CLASS_PRIORS.len() != num_classes
    {
        return placeholder_vowel_gmm();
    }

    DiagonalGmm {
        num_classes,
        num_mixtures: TRAINED_BAND_GMM_MIXTURES,
        num_features: TRAINED_BAND_GMM_FEATURES,
        means: &TRAINED_BAND_GMM_MEANS,
        inv_vars: &TRAINED_BAND_GMM_INV_VARS,
        log_weights: &TRAINED_BAND_GMM_LOG_WEIGHTS,
        class_priors: &TRAINED_BAND_GMM_CLASS_PRIORS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_generated_model_falls_back_to_placeholder() {
        let model = trained_band_vowel_gmm();
        assert_eq!(model.num_classes, crate::gmm::PLACEHOLDER_GMM_CLASSES);
        assert_eq!(model.num_features, crate::gmm::PLACEHOLDER_GMM_FEATURES);
    }
}

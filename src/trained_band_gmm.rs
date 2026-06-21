// Generated model slot for the trained 16-band vowel GMM.
// Empty arrays intentionally fall back to gmm::placeholder_vowel_gmm(), which is
// infrastructure only and not an accuracy model.

use crate::gmm::{DiagonalGmm, placeholder_vowel_gmm};

pub const GMM_MODEL_KIND_NOT_USED: u32 = 0;
pub const GMM_MODEL_KIND_PLACEHOLDER: u32 = 1;
pub const GMM_MODEL_KIND_TRAINED: u32 = 2;

pub const TRAINED_BAND_GMM_CLASSES: [&str; 0] = [];
pub const TRAINED_BAND_GMM_MIXTURES: usize = 0;
pub const TRAINED_BAND_GMM_FEATURES: usize = 0;

pub const TRAINED_BAND_GMM_MEANS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_INV_VARS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_LOG_WEIGHTS: [f32; 0] = [];
pub const TRAINED_BAND_GMM_CLASS_PRIORS: [f32; 0] = [];

pub fn trained_band_vowel_gmm_model_kind() -> u32 {
    model_kind_for_shape(
        TRAINED_BAND_GMM_CLASSES.len(),
        TRAINED_BAND_GMM_MIXTURES,
        TRAINED_BAND_GMM_FEATURES,
        TRAINED_BAND_GMM_MEANS.len(),
        TRAINED_BAND_GMM_INV_VARS.len(),
        TRAINED_BAND_GMM_LOG_WEIGHTS.len(),
        TRAINED_BAND_GMM_CLASS_PRIORS.len(),
    )
}

pub(crate) fn model_kind_for_shape(
    num_classes: usize,
    num_mixtures: usize,
    num_features: usize,
    means_len: usize,
    inv_vars_len: usize,
    log_weights_len: usize,
    class_priors_len: usize,
) -> u32 {
    let expected_params = num_classes * num_mixtures * num_features;
    let expected_mixtures = num_classes * num_mixtures;
    if num_classes == 0
        || num_mixtures == 0
        || num_features == 0
        || means_len != expected_params
        || inv_vars_len != expected_params
        || log_weights_len != expected_mixtures
        || class_priors_len != num_classes
    {
        GMM_MODEL_KIND_PLACEHOLDER
    } else {
        GMM_MODEL_KIND_TRAINED
    }
}

pub fn trained_band_vowel_gmm() -> DiagonalGmm {
    if trained_band_vowel_gmm_model_kind() != GMM_MODEL_KIND_TRAINED {
        return placeholder_vowel_gmm();
    }

    DiagonalGmm {
        num_classes: TRAINED_BAND_GMM_CLASSES.len(),
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
    fn empty_generated_model_reports_placeholder() {
        assert_eq!(
            trained_band_vowel_gmm_model_kind(),
            GMM_MODEL_KIND_PLACEHOLDER
        );
        let model = trained_band_vowel_gmm();
        assert_eq!(model.num_classes, crate::gmm::PLACEHOLDER_GMM_CLASSES);
        assert_eq!(model.num_features, crate::gmm::PLACEHOLDER_GMM_FEATURES);
    }

    #[test]
    fn generated_non_empty_model_reports_trained() {
        assert_eq!(
            model_kind_for_shape(5, 1, 16, 5 * 16, 5 * 16, 5, 5),
            GMM_MODEL_KIND_TRAINED
        );
    }
}

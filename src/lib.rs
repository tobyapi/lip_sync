pub mod features;
pub mod gmm;
pub mod lpc;
pub mod mapper;
pub mod normalization;
pub mod vowel;

use crate::mapper::{LipSyncMappedFrame, map_frame};
use crate::vowel::{
    LipSyncAnalyzer, LipSyncDebugFrame, LipSyncFrame, LipSyncOptions, LipSyncTimedCue,
    LipSyncTrainingFeatures, extract_training_features_from_pcm, lip_sync_class_from_index,
};
use libc::size_t;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_default_options(sample_rate: u32) -> LipSyncOptions {
    LipSyncOptions::for_sample_rate(sample_rate).normalized()
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_singing_options(sample_rate: u32) -> LipSyncOptions {
    LipSyncOptions::singing_preset(sample_rate).normalized()
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_extract_training_features(
    pcm: *const f32,
    len: size_t,
    options: LipSyncOptions,
    result: *mut LipSyncTrainingFeatures,
) -> bool {
    if pcm.is_null() || result.is_null() || len == 0 || options.sample_rate == 0 {
        return false;
    }

    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let features = extract_training_features_from_pcm(pcm_data, options);

    unsafe {
        *result = features;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_frame_best_class(frame: *const LipSyncFrame, result: *mut u32) -> bool {
    if frame.is_null() || result.is_null() {
        return false;
    }

    let frame = unsafe { &*frame };
    unsafe {
        *result = frame.best_class() as u32;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_frame_class_score(
    frame: *const LipSyncFrame,
    class_index: u32,
    result: *mut f32,
) -> bool {
    if frame.is_null() || result.is_null() {
        return false;
    }

    let Some(class) = lip_sync_class_from_index(class_index) else {
        return false;
    };

    let frame = unsafe { &*frame };
    unsafe {
        *result = frame.posterior[class as usize];
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_map_frame(
    frame: *const LipSyncFrame,
    mapper_kind: u32,
    result: *mut LipSyncMappedFrame,
) -> bool {
    if frame.is_null() || result.is_null() {
        return false;
    }

    let frame = unsafe { &*frame };
    let Some(mapped) = map_frame(frame, mapper_kind) else {
        return false;
    };

    unsafe {
        *result = mapped;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_create(sample_rate: u32, singing_mode: bool) -> *mut LipSyncAnalyzer {
    Box::into_raw(Box::new(LipSyncAnalyzer::new(sample_rate, singing_mode)))
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_create_with_options(options: LipSyncOptions) -> *mut LipSyncAnalyzer {
    Box::into_raw(Box::new(LipSyncAnalyzer::with_options(options)))
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_set_timed_cues(
    analyzer: *mut LipSyncAnalyzer,
    cues: *const LipSyncTimedCue,
    len: size_t,
) -> bool {
    if analyzer.is_null() {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };

    if cues.is_null() || len == 0 {
        analyzer.clear_timed_cues();
        return true;
    }

    let cues = unsafe { slice::from_raw_parts(cues, len as usize) };
    analyzer.set_timed_cues(cues);
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_clear_timed_cues(analyzer: *mut LipSyncAnalyzer) -> bool {
    if analyzer.is_null() {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    analyzer.clear_timed_cues();
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    result: *mut LipSyncFrame,
) -> bool {
    if analyzer.is_null() || pcm.is_null() || result.is_null() || len == 0 {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process(pcm_data);

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_debug(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    result: *mut LipSyncDebugFrame,
) -> bool {
    if analyzer.is_null() || pcm.is_null() || result.is_null() || len == 0 {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process_debug(pcm_data);

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_at_time(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    time_seconds: f32,
    result: *mut LipSyncFrame,
) -> bool {
    if analyzer.is_null()
        || pcm.is_null()
        || result.is_null()
        || len == 0
        || !time_seconds.is_finite()
    {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process_at_time(pcm_data, time_seconds.max(0.0));

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_at_time_debug(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    time_seconds: f32,
    result: *mut LipSyncDebugFrame,
) -> bool {
    if analyzer.is_null()
        || pcm.is_null()
        || result.is_null()
        || len == 0
        || !time_seconds.is_finite()
    {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process_at_time_debug(pcm_data, time_seconds.max(0.0));

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_mapped(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    mapper_kind: u32,
    result: *mut LipSyncMappedFrame,
) -> bool {
    if analyzer.is_null() || pcm.is_null() || result.is_null() || len == 0 {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process(pcm_data);
    let Some(mapped) = map_frame(&frame, mapper_kind) else {
        return false;
    };

    unsafe {
        *result = mapped;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_at_time_mapped(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    len: size_t,
    time_seconds: f32,
    mapper_kind: u32,
    result: *mut LipSyncMappedFrame,
) -> bool {
    if analyzer.is_null()
        || pcm.is_null()
        || result.is_null()
        || len == 0
        || !time_seconds.is_finite()
    {
        return false;
    }

    let analyzer = unsafe { &mut *analyzer };
    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };
    let frame = analyzer.process_at_time(pcm_data, time_seconds.max(0.0));
    let Some(mapped) = map_frame(&frame, mapper_kind) else {
        return false;
    };

    unsafe {
        *result = mapped;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_interleaved(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    frame_count: size_t,
    channels: u32,
    result: *mut LipSyncFrame,
) -> bool {
    if result.is_null() {
        return false;
    }
    let Some(frame) = process_interleaved(analyzer, pcm, frame_count, channels, None) else {
        return false;
    };

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_interleaved_at_time(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    frame_count: size_t,
    channels: u32,
    time_seconds: f32,
    result: *mut LipSyncFrame,
) -> bool {
    if result.is_null() {
        return false;
    }
    let Some(frame) = process_interleaved(analyzer, pcm, frame_count, channels, Some(time_seconds))
    else {
        return false;
    };

    unsafe {
        *result = frame;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_interleaved_mapped(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    frame_count: size_t,
    channels: u32,
    mapper_kind: u32,
    result: *mut LipSyncMappedFrame,
) -> bool {
    if result.is_null() {
        return false;
    }
    let Some(frame) = process_interleaved(analyzer, pcm, frame_count, channels, None) else {
        return false;
    };
    let Some(mapped) = map_frame(&frame, mapper_kind) else {
        return false;
    };

    unsafe {
        *result = mapped;
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_process_interleaved_at_time_mapped(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    frame_count: size_t,
    channels: u32,
    time_seconds: f32,
    mapper_kind: u32,
    result: *mut LipSyncMappedFrame,
) -> bool {
    if result.is_null() {
        return false;
    }
    let Some(frame) = process_interleaved(analyzer, pcm, frame_count, channels, Some(time_seconds))
    else {
        return false;
    };
    let Some(mapped) = map_frame(&frame, mapper_kind) else {
        return false;
    };

    unsafe {
        *result = mapped;
    }
    true
}

fn process_interleaved(
    analyzer: *mut LipSyncAnalyzer,
    pcm: *const f32,
    frame_count: size_t,
    channels: u32,
    time_seconds: Option<f32>,
) -> Option<LipSyncFrame> {
    if analyzer.is_null() || pcm.is_null() || frame_count == 0 || channels == 0 {
        return None;
    }
    if time_seconds.is_some_and(|time| !time.is_finite()) {
        return None;
    }

    let frame_count = frame_count as usize;
    let channels = channels as usize;
    let len = frame_count.checked_mul(channels)?;
    let interleaved = unsafe { slice::from_raw_parts(pcm, len) };
    let mono = downmix_interleaved(interleaved, frame_count, channels)?;
    let analyzer = unsafe { &mut *analyzer };

    Some(match time_seconds {
        Some(time) => analyzer.process_at_time(&mono, time.max(0.0)),
        None => analyzer.process(&mono),
    })
}

fn downmix_interleaved(
    interleaved: &[f32],
    frame_count: usize,
    channels: usize,
) -> Option<Vec<f32>> {
    if frame_count == 0 || channels == 0 || interleaved.len() < frame_count.checked_mul(channels)? {
        return None;
    }

    if channels == 1 {
        return Some(interleaved[..frame_count].to_vec());
    }

    let mut mono = Vec::with_capacity(frame_count);
    for frame in interleaved.chunks_exact(channels).take(frame_count) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }
    Some(mono)
}

#[unsafe(no_mangle)]
pub extern "C" fn lipsync_destroy(analyzer: *mut LipSyncAnalyzer) {
    if analyzer.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(analyzer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vowel::{
        LIPSYNC_FLAG_ROBUST_LOUDNESS, LIPSYNC_FLAG_TIMED_CUES, LipSyncClass, LipSyncCueKind,
    };

    #[test]
    fn lipsync_c_abi_lifecycle_processes_a_frame() {
        let analyzer = lipsync_create(44_100, false);
        assert!(!analyzer.is_null());

        let pcm = vec![0.0; 1024];
        let mut frame = LipSyncFrame::default();
        let ok = lipsync_process(analyzer, pcm.as_ptr(), pcm.len(), &mut frame);

        assert!(ok);
        assert!(frame.posterior[LipSyncClass::Rest as usize] > 0.9);

        lipsync_destroy(analyzer);
    }

    #[test]
    fn lipsync_c_abi_debug_processes_raw_fields() {
        let analyzer = lipsync_create(44_100, false);
        assert!(!analyzer.is_null());

        let pcm = vec![0.0; 1024];
        let mut frame = LipSyncDebugFrame::default();
        let ok = lipsync_process_debug(analyzer, pcm.as_ptr(), pcm.len(), &mut frame);

        assert!(ok);
        assert!(frame.frame.posterior[LipSyncClass::Rest as usize] > 0.9);
        assert!(frame.rms >= 0.0);
        assert_eq!(
            frame.band_feature_space,
            crate::vowel::LIPSYNC_FEATURE_SPACE_BANDS_16
        );
        assert_eq!(
            frame.feature_vector_space,
            crate::vowel::LIPSYNC_FEATURE_SPACE_VECTOR_31
        );
        assert!(frame.normalized_bands.iter().all(|value| value.is_finite()));
        assert!(frame.feature_vector.iter().all(|value| value.is_finite()));
        assert_eq!(frame.raw_best_vowel, -1);

        lipsync_destroy(analyzer);
    }

    #[test]
    fn lipsync_c_abi_extracts_training_features() {
        let pcm = vec![0.02; 1024];
        let options = LipSyncOptions {
            sample_rate: 16_000,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS,
            metadata_weight: 0.0,
            smoothing: 0.18,
            loudness_adaptation: 0.07,
        };
        let mut features = LipSyncTrainingFeatures::default();

        assert!(lipsync_extract_training_features(
            pcm.as_ptr(),
            pcm.len(),
            options,
            &mut features
        ));
        assert_eq!(features.sample_rate, 16_000);
        assert_eq!(
            features.band_feature_space,
            crate::vowel::LIPSYNC_FEATURE_SPACE_BANDS_16
        );
        assert_eq!(
            features.feature_vector_space,
            crate::vowel::LIPSYNC_FEATURE_SPACE_VECTOR_31
        );
        assert!(
            features
                .normalized_bands
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            features
                .feature_vector
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(!lipsync_extract_training_features(
            std::ptr::null(),
            pcm.len(),
            options,
            &mut features
        ));
    }
    #[test]
    fn lipsync_c_abi_applies_timed_cues_at_time() {
        let analyzer = lipsync_create_with_options(LipSyncOptions {
            sample_rate: 44_100,
            flags: LIPSYNC_FLAG_ROBUST_LOUDNESS | LIPSYNC_FLAG_TIMED_CUES,
            metadata_weight: 0.85,
            smoothing: 0.0,
            loudness_adaptation: 0.07,
        });
        assert!(!analyzer.is_null());

        let cue = LipSyncTimedCue {
            start_seconds: 0.0,
            end_seconds: 1.0,
            class_index: LipSyncClass::E as u32,
            weight: 1.0,
            kind: LipSyncCueKind::LyricTiming as u32,
        };
        assert!(lipsync_set_timed_cues(analyzer, &cue, 1));

        let pcm = vec![0.02; 2048];
        let mut frame = LipSyncFrame::default();
        assert!(lipsync_process_at_time(
            analyzer,
            pcm.as_ptr(),
            pcm.len(),
            0.5,
            &mut frame
        ));
        assert!(
            frame.posterior[LipSyncClass::E as usize]
                > frame.posterior[LipSyncClass::Rest as usize]
        );

        assert!(lipsync_clear_timed_cues(analyzer));
        lipsync_destroy(analyzer);
    }
    #[test]
    fn lipsync_c_abi_frame_helpers_and_mapper_work() {
        let mut frame = LipSyncFrame::default();
        frame.posterior = [0.04, 0.02, 0.82, 0.03, 0.02, 0.03, 0.02, 0.01, 0.01];
        frame.jaw_open = 0.7;

        let mut class_index = 0;
        assert!(lipsync_frame_best_class(&frame, &mut class_index));
        assert_eq!(class_index, LipSyncClass::A as u32);

        let mut mapped = LipSyncMappedFrame::default();
        assert!(lipsync_map_frame(
            &frame,
            crate::mapper::LIPSYNC_MAPPER_VRM,
            &mut mapped
        ));
        assert_eq!(mapped.best_class, LipSyncClass::A as u32);
        assert!(mapped.aa > mapped.ih);
        assert!(!lipsync_map_frame(&frame, 99, &mut mapped));
    }

    #[test]
    fn lipsync_c_abi_processes_interleaved_pcm() {
        let analyzer = lipsync_create(16_000, false);
        assert!(!analyzer.is_null());

        let mut interleaved = Vec::with_capacity(2_048);
        for _ in 0..1_024 {
            interleaved.push(0.01);
            interleaved.push(0.03);
        }

        let mut frame = LipSyncFrame::default();
        assert!(lipsync_process_interleaved(
            analyzer,
            interleaved.as_ptr(),
            1_024,
            2,
            &mut frame
        ));
        assert!(frame.posterior[LipSyncClass::Rest as usize] >= 0.0);

        let mut mapped = LipSyncMappedFrame::default();
        assert!(lipsync_process_interleaved_mapped(
            analyzer,
            interleaved.as_ptr(),
            1_024,
            2,
            crate::mapper::LIPSYNC_MAPPER_GENERIC,
            &mut mapped
        ));
        assert!(mapped.jaw_open >= 0.0);
        assert!(!lipsync_process_interleaved(
            analyzer,
            interleaved.as_ptr(),
            1_024,
            0,
            &mut frame
        ));

        lipsync_destroy(analyzer);
    }
}

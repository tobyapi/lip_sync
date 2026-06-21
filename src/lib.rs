pub mod lpc;
pub mod vowel;

use crate::vowel::{
    LipSyncAnalyzer, LipSyncFrame, LipSyncOptions, LipSyncTimedCue, Vowel, recognize_vowel_from_pcm,
};
use libc::size_t;
use std::slice;

#[unsafe(no_mangle)]
pub extern "C" fn recognize_vowel(
    pcm: *const f32,
    len: size_t,
    sample_rate: u32,
    result: *mut Vowel,
) -> bool {
    if pcm.is_null() || result.is_null() || len == 0 || sample_rate == 0 {
        return false;
    }

    let pcm_data = unsafe { slice::from_raw_parts(pcm, len as usize) };

    if let Some(vowel) = recognize_vowel_from_pcm(pcm_data, sample_rate) {
        unsafe {
            *result = vowel;
        }
        true
    } else {
        false
    }
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
    use std::ptr;

    #[test]
    fn recognize_vowel_ffi_rejects_nulls() {
        let mut vowel = Vowel::A;
        assert!(!recognize_vowel(ptr::null(), 1024, 44_100, &mut vowel));

        let pcm = vec![0.0; 1024];
        assert!(!recognize_vowel(
            pcm.as_ptr(),
            pcm.len(),
            44_100,
            ptr::null_mut()
        ));
    }

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

        let pcm = vec![0.02; 1024];
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
}

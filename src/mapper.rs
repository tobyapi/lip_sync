use crate::vowel::{LipSyncClass, LipSyncFrame};

pub const LIPSYNC_MAPPER_GENERIC: u32 = 0;
pub const LIPSYNC_MAPPER_VRM: u32 = 1;
pub const LIPSYNC_MAPPER_ARKIT: u32 = 2;
pub const LIPSYNC_MAPPER_METAHUMAN: u32 = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LipSyncMappedFrame {
    pub kind: u32,
    pub best_class: u32,
    pub confidence: f32,
    pub jaw_open: f32,
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
    pub mouth_close: f32,
    pub mouth_funnel: f32,
    pub mouth_pucker: f32,
    pub mouth_wide: f32,
    pub mouth_smile_left: f32,
    pub mouth_smile_right: f32,
    pub mouth_lower_down_left: f32,
    pub mouth_lower_down_right: f32,
    pub mouth_upper_up_left: f32,
    pub mouth_upper_up_right: f32,
    pub mouth_press_left: f32,
    pub mouth_press_right: f32,
    pub fricative: f32,
}

impl Default for LipSyncMappedFrame {
    fn default() -> Self {
        Self {
            kind: LIPSYNC_MAPPER_GENERIC,
            best_class: LipSyncClass::Rest as u32,
            confidence: 1.0,
            jaw_open: 0.0,
            aa: 0.0,
            ih: 0.0,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
            mouth_close: 0.0,
            mouth_funnel: 0.0,
            mouth_pucker: 0.0,
            mouth_wide: 0.0,
            mouth_smile_left: 0.0,
            mouth_smile_right: 0.0,
            mouth_lower_down_left: 0.0,
            mouth_lower_down_right: 0.0,
            mouth_upper_up_left: 0.0,
            mouth_upper_up_right: 0.0,
            mouth_press_left: 0.0,
            mouth_press_right: 0.0,
            fricative: 0.0,
        }
    }
}

pub fn map_frame(frame: &LipSyncFrame, kind: u32) -> Option<LipSyncMappedFrame> {
    if !matches!(
        kind,
        LIPSYNC_MAPPER_GENERIC
            | LIPSYNC_MAPPER_VRM
            | LIPSYNC_MAPPER_ARKIT
            | LIPSYNC_MAPPER_METAHUMAN
    ) {
        return None;
    }

    let rest = frame.posterior[LipSyncClass::Rest as usize].clamp(0.0, 1.0);
    let closed = frame.posterior[LipSyncClass::Closed as usize].clamp(0.0, 1.0);
    let a = frame.posterior[LipSyncClass::A as usize].clamp(0.0, 1.0);
    let i = frame.posterior[LipSyncClass::I as usize].clamp(0.0, 1.0);
    let u = frame.posterior[LipSyncClass::U as usize].clamp(0.0, 1.0);
    let e = frame.posterior[LipSyncClass::E as usize].clamp(0.0, 1.0);
    let o = frame.posterior[LipSyncClass::O as usize].clamp(0.0, 1.0);
    let fricative = frame.posterior[LipSyncClass::Fricative as usize].clamp(0.0, 1.0);

    let vowel_mass = a + i + u + e + o;
    let activity = (1.0 - rest).clamp(0.0, 1.0);
    let closed_damping = (1.0 - closed * 0.85).clamp(0.0, 1.0);
    let vowel_scale = if vowel_mass > f32::EPSILON {
        (activity / vowel_mass).clamp(0.0, 1.4) * closed_damping
    } else {
        0.0
    };

    let aa = clamp01(a * vowel_scale);
    let ih = clamp01(i * vowel_scale);
    let ou = clamp01(u * vowel_scale);
    let ee = clamp01(e * vowel_scale);
    let oh = clamp01(o * vowel_scale);
    let rounded = clamp01(ou * 0.85 + oh * 0.75);
    let wide = clamp01(ee * 0.65 + ih * 0.55);
    let open = clamp01(frame.jaw_open);
    let mouth_close = match kind {
        LIPSYNC_MAPPER_VRM => 0.0,
        _ => clamp01(closed * 0.95 + rest * 0.18),
    };

    let mut mapped = LipSyncMappedFrame {
        kind,
        best_class: frame.best_class() as u32,
        confidence: frame.best_class_score(),
        jaw_open: open,
        aa,
        ih,
        ou,
        ee,
        oh,
        mouth_close,
        mouth_funnel: clamp01(rounded + fricative * 0.08),
        mouth_pucker: clamp01(ou * 0.75 + oh * 0.35),
        mouth_wide: wide,
        mouth_smile_left: clamp01(wide * 0.45),
        mouth_smile_right: clamp01(wide * 0.45),
        mouth_lower_down_left: clamp01(open * (0.55 + aa * 0.30 + oh * 0.12)),
        mouth_lower_down_right: clamp01(open * (0.55 + aa * 0.30 + oh * 0.12)),
        mouth_upper_up_left: clamp01(aa * 0.22 + ee * 0.18),
        mouth_upper_up_right: clamp01(aa * 0.22 + ee * 0.18),
        mouth_press_left: clamp01(closed * 0.70),
        mouth_press_right: clamp01(closed * 0.70),
        fricative: clamp01(fricative * activity.max(0.35)),
    };

    if kind == LIPSYNC_MAPPER_METAHUMAN {
        mapped.mouth_funnel = clamp01(mapped.mouth_funnel * 0.9);
        mapped.mouth_pucker = clamp01(mapped.mouth_pucker * 0.9);
        mapped.mouth_wide = clamp01(mapped.mouth_wide * 0.85);
        mapped.mouth_smile_left = clamp01(mapped.mouth_smile_left * 0.75);
        mapped.mouth_smile_right = clamp01(mapped.mouth_smile_right * 0.75);
    }

    Some(mapped)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_vrm_visemes_from_vowel_posterior() {
        let mut frame = LipSyncFrame::default();
        frame.posterior = [0.05, 0.02, 0.82, 0.03, 0.02, 0.03, 0.02, 0.0, 0.01];
        frame.jaw_open = 0.75;

        let mapped = map_frame(&frame, LIPSYNC_MAPPER_VRM).unwrap();

        assert_eq!(mapped.best_class, LipSyncClass::A as u32);
        assert!(mapped.aa > 0.8);
        assert!(mapped.ih < 0.1);
        assert_eq!(mapped.mouth_close, 0.0);
    }

    #[test]
    fn maps_closed_to_arkit_mouth_close() {
        let mut frame = LipSyncFrame::default();
        frame.posterior = [0.02, 0.86, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04];

        let mapped = map_frame(&frame, LIPSYNC_MAPPER_ARKIT).unwrap();

        assert_eq!(mapped.best_class, LipSyncClass::Closed as u32);
        assert!(mapped.mouth_close > 0.8);
        assert!(mapped.aa < 0.1);
    }
}

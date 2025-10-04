use crate::lpc;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Vowel {
    A,
    I,
    U,
    E,
    O,
}

#[derive(Debug)]
pub struct Formant {
    pub frequency: f32,
    pub magnitude: f32,
}

/// スペクトル包絡からフォルマントを検出します。
///
/// スペクトルのピーク（極大値）を探すことで、フォルマント周波数を特定します。
///
/// # Arguments
/// * `spectrum` - スペクトル包絡のベクトル。
/// * `sample_rate` - 元の音声のサンプリング周波数。
/// * `max_formants` - 検出するフォルマントの最大数。
///
/// # Returns
/// * `Vec<Formant>` - 検出されたフォルマントのベクトル。周波数と振幅の情報を含みます。
pub fn find_formants(
    spectrum: &[f32],
    sample_rate: u32,
    max_formants: usize,
) -> Vec<Formant> {
    let mut formants = Vec::new();
    let fft_size = spectrum.len();

    // Limit search to 5000 Hz
    let max_freq_index = (5000.0 / (sample_rate as f32 / fft_size as f32)) as usize;

    for i in 1..max_freq_index.min(fft_size / 2 - 1) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
            let freq = i as f32 * sample_rate as f32 / fft_size as f32;
            formants.push(Formant {
                frequency: freq,
                magnitude: spectrum[i],
            });
        }
    }

    // Sort by magnitude
    formants.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap());

    // Keep top N
    formants.truncate(max_formants);
    // Sort by frequency
    formants.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());

    formants
}

/// 第1フォルマントと第2フォルマントから母音を判定します。
///
/// 事前に定義された日本語の各母音のF1/F2周波数の中心値と、
/// 検出されたフォルマントの周波数とのユークリッド距離を計算し、
/// 最も距離が近い母音を判定結果とします。
///
/// # Arguments
/// * `formants` - 検出されたフォルマントのベクトル。
///
/// # Returns
/// * `Option<Vowel>` - フォルマントが2つ以上検出された場合、最も近い母音を返します。
///   そうでなければ `None` を返します。
pub fn recognize_vowel(formants: &[Formant]) -> Option<Vowel> {
    if formants.len() < 2 {
        return None;
    }

    let f1 = formants[0].frequency;
    let f2 = formants[1].frequency;

    // 日本語母音のF1/F2周波数の典型値（成人男性の例）
    //let vowel_centers = [
    //    (700.0, 1200.0, Vowel::A),
    //    (300.0, 2300.0, Vowel::I),
    //    (300.0, 1300.0, Vowel::U),
    //    (500.0, 1900.0, Vowel::E),
    //    (500.0, 900.0, Vowel::O),
    //];

    // 日本語母音のF1/F2周波数の典型値（成人女性の例）
    let vowel_centers = [
        (1000.0, 1600.0, Vowel::A),
        (300.0, 2800.0, Vowel::I),
        (310.0, 1300.0, Vowel::U),
        (500.0, 2500.0, Vowel::E),
        (500.0, 900.0, Vowel::O),
    ];

    let mut min_dist = f32::MAX;
    let mut closest_vowel = None;

    for (center_f1, center_f2, vowel) in vowel_centers.iter() {
        let dist = ((f1 - center_f1).powi(2) + (f2 - center_f2).powi(2)).sqrt();
        if dist < min_dist {
            min_dist = dist;
            closest_vowel = Some(vowel);
        }
    }

    closest_vowel.cloned()
}

/// Analyzes a PCM audio chunk and recognizes the most likely vowel.
///
/// This function encapsulates the entire process of vowel recognition from a raw audio signal frame.
/// It performs LPC analysis, formant detection, and vowel classification.
///
/// # Arguments
/// * `pcm_data` - A slice of f32 representing the audio chunk. Recommended lengths are 512, 1024, or 2048.
/// * `sample_rate` - The sample rate of the audio data.
///
/// # Returns
/// * `Option<Vowel>` - The recognized vowel, or `None` if recognition fails (e.g., no formants detected).
pub fn recognize_vowel_from_pcm(pcm_data: &[f32], sample_rate: u32) -> Option<Vowel> {
    let order = 24; // LPC analysis order
    let fft_size = 1024; // FFT size for spectral envelope calculation

    if pcm_data.is_empty() {
        return None;
    }

    let mut signal_chunk = pcm_data.to_vec();
    lpc::hamming_window(&mut signal_chunk);

    let mut acf = lpc::autocorrelate(&signal_chunk);
    let acf0 = acf[0];
    if acf0 > 0.0 {
        for val in &mut acf {
            *val /= acf0;
        }
    } else {
        // If energy is zero, no signal is present.
        return None;
    }

    if let Some((alpha, err)) = lpc::levinson_durbin(&acf, order) {
        let gain = err * acf0;
        
        let spectral_envelope = lpc::lpc_to_spectral_envelope(&alpha, gain, fft_size);
        let formants = find_formants(&spectral_envelope, sample_rate, 5);

        recognize_vowel(&formants)
    } else {
        None
    }
}

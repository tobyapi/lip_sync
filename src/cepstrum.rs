
use std::collections::HashMap;
use num_complex::Complex;
use rustfft::FftPlanner;

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

/// 音声信号の自己相関を計算します。
///
/// 自己相関は、信号が時間的にどれだけ自分自身に似ているかを示す尺度です。
/// LPC（線形予測符号化）分析の最初のステップとして、この計算が必要になります。
///
/// # Arguments
/// * `signal` - 入力となる音声信号のスライス。値はf32型で正規化されていることが望ましいです。
///
/// # Returns
/// * `Vec<f32>` - 計算された自己相関値のベクトル。
pub fn autocorrelate(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut acf = vec![0.0; n];
    for i in 0..n {
        for j in 0..(n - i) {
            acf[i] += signal[j] * signal[j + i];
        }
    }
    acf
}

/// 信号にハミング窓を適用します。
///
/// 窓関数は、信号の短いフレームを切り出す際に、その両端が不連続になること（スペクトル漏れ）を防ぐために使います。
/// ハミング窓は、音声分析で非常によく使われる窓関数の一つです。
///
/// # Arguments
/// * `signal` - 窓を適用する信号のスライス。この関数は信号を直接変更します。
pub fn hamming_window(signal: &mut [f32]) {
    let n = signal.len();
    for i in 0..n {
        signal[i] *= 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
    }
}

/// レビンソン・ダービン法を用いて、自己相関値からLPC係数を計算します。
///
/// このアルゴリズムは、効率的にLPC係数（alpha）と予測誤差（err）を求めることができます。
/// LPC係数は、ある時点の音声サンプルが、それ以前のサンプルからどのように予測できるかを示します。
///
/// # Arguments
/// * `acf` - 自己相関値のベクトル。`acf[0]`は信号のエネルギーに相当します。正規化されていることが望ましいです。
/// * `order` - LPC分析の次数。どれだけ過去のサンプルを予測に使うかを決めます。
///
/// # Returns
/// * `Option<(Vec<f32>, f32)>` - 計算が成功した場合、LPC係数（alpha）のベクトルと最終的な予測誤差（err）のタプルを返します。
///   予測誤差が0になるなど、計算が不安定な場合は `None` を返します。
pub fn levinson_durbin(acf: &[f32], order: usize) -> Option<(Vec<f32>, f32)> {
    if acf.is_empty() {
        return None;
    }
    let mut alpha = vec![0.0; order + 1];
    let mut err = acf[0];
    if err == 0.0 {
        return None;
    }

    alpha[0] = 1.0;

    for i in 1..=order {
        let mut k = -acf[i];
        for j in 1..i {
            k -= alpha[j] * acf[i - j];
        }
        k /= err;

        let old_alpha = alpha.clone();
        alpha[i] = k;
        for j in 1..i {
            alpha[j] = old_alpha[j] + k * old_alpha[i - j];
        }
        err *= 1.0 - k * k;
        if err == 0.0 {
            return None;
        }
    }

    Some((alpha, err))
}

/// LPC係数からケプストラム係数を計算します。
///
/// ケプストラムは、音声のスペクトル包絡（声道の特性）を表現する特徴量です。
/// この関数は、LPC係数から再帰的な計算式で直接ケプストラムに変換します。
///
/// # Arguments
/// * `alpha` - LPC係数のベクトル。`levinson_durbin`から得られたものです。
/// * `err` - 予測誤差のゲイン。これも`levinson_durbin`から得られます。
/// * `num_coeffs` - 計算するケプストラム係数の数。
///
/// # Returns
/// * `Vec<f32>` - 計算されたケプストラム係数のベクトル。
pub fn lpc_to_cepstrum(alpha: &[f32], err: f32, num_coeffs: usize) -> Vec<f32> {
    let p = alpha.len() - 1;
    let mut c = vec![0.0; num_coeffs];

    if num_coeffs > 0 {
        if err > 0.0 {
            c[0] = err.ln();
        } else {
            c[0] = f32::NEG_INFINITY;
        }
    }

    let a = &alpha;

    for m in 1..num_coeffs {
        let mut sum = 0.0;
        for k in 1..m {
            sum += a[k] * c[m - k];
        }
        if m <= p {
            c[m] = -a[m] - sum;
        } else {
            c[m] = -sum;
        }
    }

    c
}

/// LPC係数からスペクトル包絡を計算します。
///
/// LPC係数のフーリエ変換の逆数をとることで、スペクトル包絡を計算します。
///
/// # Arguments
/// * `alpha` - LPC係数のベクトル。
/// * `gain` - 予測誤差のゲイン。
/// * `fft_size` - FFTのサイズ。
///
/// # Returns
/// * `Vec<f32>` - 計算されたスペクトル包絡（対数マグニチュード）。
pub fn lpc_to_spectral_envelope(
    alpha: &[f32],
    gain: f32,
    fft_size: usize,
) -> Vec<f32> {
    let mut a = vec![0.0; fft_size];
    a[0..alpha.len()].copy_from_slice(alpha);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer: Vec<Complex<f32>> = a
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    fft.process(&mut buffer);

    let log_gain = if gain > 0.0 { gain.ln() } else { f32::NEG_INFINITY };

    buffer
        .iter()
        .map(|c| log_gain - 2.0 * c.norm().ln())
        .collect()
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
    hamming_window(&mut signal_chunk);

    let mut acf = autocorrelate(&signal_chunk);
    let acf0 = acf[0];
    if acf0 > 0.0 {
        for val in &mut acf {
            *val /= acf0;
        }
    } else {
        // If energy is zero, no signal is present.
        return None;
    }

    if let Some((alpha, err)) = levinson_durbin(&acf, order) {
        let gain = err * acf0;
        
        let spectral_envelope = lpc_to_spectral_envelope(&alpha, gain, fft_size);
        let formants = find_formants(&spectral_envelope, sample_rate, 5);

        recognize_vowel(&formants)
    } else {
        None
    }
}

/// ケプストラム係数にリフタリングを適用します。
///
/// 低域ケプストラム係数のみを保持し、高域ケプストラム係数をゼロにすることで、
/// スペクトル包絡を平滑化し、音源情報（ピッチなど）を除去します。
///
/// # Arguments
/// * `cepstrum` - ケプストラム係数のベクトル。
/// * `lifter_length` - 保持する低域ケプストラム係数の数。
///
/// # Returns
/// * `Vec<f32>` - リフタリングが適用されたケプストラム係数のベクトル。
pub fn lifter_cepstrum(mut cepstrum: Vec<f32>, lifter_length: usize) -> Vec<f32> {
    for i in lifter_length..cepstrum.len() {
        cepstrum[i] = 0.0;
    }
    cepstrum
}

/// リフタリングされたケプストラム係数からLPC係数を再構築します。
///
/// # Arguments
/// * `cepstrum` - リフタリングが適用されたケプストラム係数のベクトル (c[0]はゲインの対数)。
/// * `lpc_order` - 再構築するLPC係数の次数。
///
/// # Returns
/// * `Vec<f32>` - 再構築されたLPC係数のベクトル (a[0]は1.0)。
pub fn cepstrum_to_lpc(cepstrum: &[f32], lpc_order: usize) -> Vec<f32> {
    let mut a = vec![0.0; lpc_order + 1];
    a[0] = 1.0; // a[0] is always 1.0 for LPC coefficients

    // The cepstrum coefficients are c[1]...c[lpc_order] for the direct conversion.
    // c[0] is related to the gain, not directly used in a[m] calculation for m > 0.

    for m in 1..=lpc_order {
        let mut sum = 0.0;
        for k in 1..m {
            // a_m = -c_m - sum_{k=1}^{m-1} (k/m) * a_k * c_{m-k}
            // Ensure indices are within bounds for cepstrum and a
            if k < a.len() && (m - k) < cepstrum.len() {
                sum += (k as f32 / m as f32) * a[k] * cepstrum[m - k];
            }
        }
        if m < cepstrum.len() {
            a[m] = -cepstrum[m] - sum;
        } else {
            // If m is beyond the available cepstrum coefficients (due to liftering),
            // assume c_m is 0.
            a[m] = -sum;
        }
    }
    a
}

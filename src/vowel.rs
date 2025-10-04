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
pub fn find_formants(spectrum: &[f32], sample_rate: u32, max_formants: usize) -> Vec<Formant> {
    let mut formants = Vec::new();
    let fft_size = spectrum.len();

    // 探索範囲を5000 Hzに制限
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

    // 振幅でソート
    formants.sort_by(|a, b| b.magnitude.partial_cmp(&a.magnitude).unwrap());

    // 上位N個を保持
    formants.truncate(max_formants);
    // 周波数でソート
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
pub fn find_vowel(formants: &[Formant]) -> Option<Vowel> {
    if formants.len() < 2 {
        return None;
    }

    let formant_point = Point { x: formants[0].frequency, y: formants[1].frequency };

    const VOWEL_A: &[Point] = &[
        Point { x: 500.0, y: 0.0 },
        Point { x: 1500.0, y: 0.0 },
        Point { x: 1500.0, y: 2800.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 800.0, y: 2700.0 },
        Point { x: 580.0, y: 2100.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 600.0, y: 1500.0 },
    ];
    const VOWEL_I: &[Point] = &[
        Point { x: 0.0, y: 2100.0 },
        Point { x: 500.0, y: 2100.0 },
        Point { x: 500.0, y: 2800.0 },
        Point { x: 800.0, y: 2700.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 500.0, y: 3200.0 },
        Point { x: 550.0, y: 5000.0 },
        Point { x: 0.0, y: 5000.0 },
    ];
    const VOWEL_U: &[Point] = &[
        Point { x: 0.0, y: 800.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 580.0, y: 2100.0 },
        Point { x: 500.0, y: 2100.0 },
        Point { x: 0.0, y: 2100.0 },
    ];
    const VOWEL_E: &[Point] = &[
        Point { x: 500.0, y: 2100.0 },
        Point { x: 580.0, y: 2100.0 },
        Point { x: 800.0, y: 2700.0 },
        Point { x: 500.0, y: 2800.0 },
    ];
    const VOWEL_O_1: &[Point] = &[
        Point { x: 0.0, y: 0.0 },
        Point { x: 500.0, y: 0.0 },
        Point { x: 650.0, y: 1500.0 },
        Point { x: 600.0, y: 1900.0 },
        Point { x: 0.0, y: 800.0 },
    ];
    const VOWEL_O_2: &[Point] = &[
        Point { x: 500.0, y: 3200.0 },
        Point { x: 700.0, y: 3100.0 },
        Point { x: 1500.0, y: 2800.0 },
        Point { x: 1500.0, y: 5000.0 },
        Point { x: 550.0, y: 5000.0 },
    ];

    let vowel_polygons: &[(&[Point], Vowel)] = &[
        (VOWEL_A, Vowel::A),
        (VOWEL_I, Vowel::I),
        (VOWEL_U, Vowel::U),
        (VOWEL_E, Vowel::E),
        (VOWEL_O_1, Vowel::O),
        (VOWEL_O_2, Vowel::O),
    ];

    for (polygon, vowel) in vowel_polygons {
        if is_inside_convex_polygon(formant_point, polygon) {
            return Some(*vowel);
        }
    }

    None
}

/// PCM音声チャンクを分析し、最も可能性の高い母音を認識します。
///
/// この関数は、生の音声信号フレームからの母音認識プロセス全体をカプセル化します。
/// LPC分析、フォルマント検出、母音分類を実行します。
///
/// # 引数
/// * `pcm_data` - 音声チャンクを表すf32のスライス。推奨される長さは512、1024、または2048です。
/// * `sample_rate` - オーディオデータのサンプルレート。
///
/// # 戻り値
/// * `Option<Vowel>` - 認識された母音。認識に失敗した場合（例：フォルマントが検出されない）は`None`。
pub fn recognize_vowel_from_pcm(pcm_data: &[f32], sample_rate: u32) -> Option<Vowel> {
    let order = 24; // LPC分析の次数
    let fft_size = 1024; // スペクトル包絡計算のためのFFTサイズ

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
        // エネルギーがゼロの場合、信号は存在しません。
        return None;
    }

    if let Some((alpha, err)) = lpc::levinson_durbin(&acf, order) {
        let gain = err * acf0;

        let spectral_envelope = lpc::lpc_to_spectral_envelope(&alpha, gain, fft_size);
        let formants = find_formants(&spectral_envelope, sample_rate, 5);

        find_vowel(&formants)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// 凸多角形が反時計回りに与えられた場合に、指定された点がその多角形の内部にあるかどうかを判定します。
///
/// # 引数
/// * `point` - 判定対象の点。
/// * `polygon` - 凸多角形の頂点の配列。頂点は反時計回りに並んでいる必要があります。
///
/// # 戻り値
/// * 点が多角形の内部または境界上にある場合は `true`、そうでない場合は `false` を返します。
pub fn is_inside_convex_polygon(point: Point, polygon: &[Point]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    for i in 0..polygon.len() {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % polygon.len()];

        let edge_dx = p2.x - p1.x;
        let edge_dy = p2.y - p1.y;
        let point_dx = point.x - p1.x;
        let point_dy = point.y - p1.y;

        let cross_product = edge_dx * point_dy - edge_dy * point_dx;

        if cross_product < 0.0 {
            // If the point is to the "right" of any edge, it's outside.
            return false;
        }
    }

    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_inside_convex_polygon() {
        let square = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 2.0, y: 0.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 0.0, y: 2.0 },
        ];

        // Point inside
        assert!(is_inside_convex_polygon(Point { x: 1.0, y: 1.0 }, &square));

        // Point on edge
        assert!(is_inside_convex_polygon(Point { x: 1.0, y: 0.0 }, &square));
        assert!(is_inside_convex_polygon(Point { x: 2.0, y: 1.0 }, &square));

        // Point on vertex
        assert!(is_inside_convex_polygon(Point { x: 0.0, y: 0.0 }, &square));

        // Point outside
        assert!(!is_inside_convex_polygon(Point { x: -1.0, y: 1.0 }, &square));
        assert!(!is_inside_convex_polygon(Point { x: 1.0, y: 3.0 }, &square));
        assert!(!is_inside_convex_polygon(Point { x: 3.0, y: 1.0 }, &square));

        // Test with a non-square polygon
        let triangle = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 3.0, y: 1.0 },
            Point { x: 1.0, y: 3.0 },
        ];
        assert!(is_inside_convex_polygon(Point { x: 1.5, y: 1.5 }, &triangle));
        assert!(!is_inside_convex_polygon(Point { x: 0.5, y: 2.5 }, &triangle));
    }
}

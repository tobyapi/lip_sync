use num_complex::Complex;
use rustfft::FftPlanner;

pub fn perform_fft(p_c_m: &[f32]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(p_c_m.len());

    let mut buffer: Vec<Complex<f32>> = p_c_m
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    fft.process(&mut buffer);

    buffer
}

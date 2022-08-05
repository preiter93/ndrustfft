//! Perform 1-dimensional Fourier Transform with custom
//! normalization.
//!
//! cargo run --example fft_norm
use ndarray::{array, Array1};
use ndrustfft::{ndfft, ndifft, Complex, FftHandler, Normalization};

fn main() {
    let n = 3;

    // Init arrays
    let v: Array1<Complex<f64>> = array![1., 2., 3.].mapv(|x| Complex::new(x, x));
    let mut vhat: Array1<Complex<f64>> = Array1::zeros(n);
    let mut v2: Array1<Complex<f64>> = Array1::zeros(n);

    // define custom normalization
    fn custom_norm(data: &mut [Complex<f64>]) {
        let n = 2. / data.len() as f64;
        for d in data.iter_mut() {
            *d = *d * n;
        }
    }
    let norm = Normalization::Custom(custom_norm);

    // Init handler
    let mut handler = FftHandler::<f64>::new(n).normalization(norm);

    // Perform transforms
    ndfft(&v, &mut vhat, &mut handler, 0);
    ndifft(&vhat, &mut v2, &mut handler, 0);
}

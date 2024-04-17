//! Perform 1-dimensional Fourier Transform.
//!
//! cargo run --example fft1
use ndarray::{array, Array1};
use ndrustfft::{ndfft, ndifft};
use ndrustfft::{Complex, FftHandler};

fn main() {
    let mut x = array![1., 2., 3.].mapv(|x| Complex::new(x, x));
    let mut xhat = Array1::zeros(3);
    let mut handler = FftHandler::<f64>::new(3);
    ndfft(&x, &mut xhat, &mut handler, 0);
    ndifft(&xhat, &mut x, &mut handler, 0);
    println!("xhat: {xhat}");
    println!("x: {x}");
}

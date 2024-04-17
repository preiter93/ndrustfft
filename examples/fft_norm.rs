//! Perform 1-dimensional Fourier Transform with
//! different normalizations
//!
//! cargo run --example fft_norm
use ndarray::{array, Array1};
use ndrustfft::{ndfft, ndifft, Complex, FftHandler, Normalization};

fn main() {
    let n = 3;
    let v: Array1<Complex<f64>> = array![1., 2., 3.].mapv(|x| Complex::new(x, x));
    let mut vhat: Array1<Complex<f64>> = Array1::zeros(n);
    let mut v2: Array1<Complex<f64>> = Array1::zeros(n);

    println!("{:?}", v); // 1+1.j, 2+2.j, 3+3.j

    // Default normalization
    let mut handler = FftHandler::<f64>::new(n).normalization(Normalization::Default);
    ndfft(&v.clone(), &mut vhat, &mut handler, 0);
    ndifft(&vhat, &mut v2, &mut handler, 0);
    println!("{:?}", v2); // 1+1.j, 2+2.j, 3+3.j

    // No normalization
    let mut handler = FftHandler::<f64>::new(n).normalization(Normalization::None);
    ndfft(&v.clone(), &mut vhat, &mut handler, 0);
    ndifft(&vhat, &mut v2, &mut handler, 0);
    println!("{:?}", v2); // 3+3.j, 6+6.j, 9+9.j

    // Custom normalization
    let mut handler = FftHandler::<f64>::new(n).normalization(Normalization::Custom(my_norm));
    ndfft(&v.clone(), &mut vhat, &mut handler, 0);
    ndifft(&vhat, &mut v2, &mut handler, 0);
    println!("{:?}", v2); // 2+2.j, 4+4.j, 6+6.j
}

fn my_norm(data: &mut [Complex<f64>]) {
    let n = 2. / data.len() as f64;
    for d in data.iter_mut() {
        *d = *d * n;
    }
}

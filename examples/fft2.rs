//! Perform 2-dimensional Fourier Transform.
//!
//! cargo run --example fft2
use ndarray::{array, Array2, Zip};
use ndrustfft::{ndfft, ndifft};
use ndrustfft::{Complex, FftHandler};
// // Use parallel transforms:
// use ndrustfft::{ndfft_par as ndfft, ndifft_par as ndifft};

fn main() {
    let (nx, ny) = (3, 3);

    // Init arrays
    let v: Array2<Complex<f64>> =
        array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],].mapv(|x| Complex::new(x, x));
    let mut vhat: Array2<Complex<f64>> = Array2::zeros((nx, ny));

    // Init handlers
    let mut handler_ax0 = FftHandler::<f64>::new(nx);
    let mut handler_ax1 = FftHandler::<f64>::new(ny);

    // Transform
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((nx, ny));
        ndfft(&v, &mut work, &mut handler_ax1, 1);
        ndfft(&work, &mut vhat, &mut handler_ax0, 0);
    }

    // Assert with numpys solution (rounded to five digits)
    let numpy_vhat = array![
        [
            Complex::new(45., 45.),
            Complex::new(-7.09808, -1.90192),
            Complex::new(-1.90192, -7.09808)
        ],
        [
            Complex::new(-21.29423, -5.70577),
            Complex::new(0., 0.),
            Complex::new(0., 0.)
        ],
        [
            Complex::new(-5.70577, -21.29423),
            Complex::new(0., 0.),
            Complex::new(0., 0.)
        ],
    ];
    Zip::from(&vhat).and(&numpy_vhat).for_each(|&x, &y| {
        if (x.re - y.re).abs() > 1e-4 || (x.im - y.im).abs() > 1e-4 {
            panic!("Large difference of values, got {} expected {}.", x, y)
        };
    });

    // Inverse Transform
    let mut v_new: Array2<Complex<f64>> = Array2::zeros((nx, ny));
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((nx, ny));
        ndifft(&vhat, &mut work, &mut handler_ax0, 0);
        ndifft(&work, &mut v_new, &mut handler_ax1, 1);
    }

    // Assert with original array
    Zip::from(&v_new).and(&v).for_each(|&x, &y| {
        if (x.re - y.re).abs() > 1e-4 || (x.im - y.im).abs() > 1e-4 {
            panic!("Large difference of values, got {} expected {}.", x, y)
        };
    });
}

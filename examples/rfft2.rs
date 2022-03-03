//! Perform 2-dimensional Fourier Transform.
//!
//! For real-to-complex transforms, the real transform
//! is performed over the last axis, whie the remaining
//! transforms are complex.
//!
//! cargo run --example rfft2
use ndarray::{array, Array2, Zip};
use ndrustfft::{ndfft, ndfft_r2c, ndifft, ndifft_r2c};
use ndrustfft::{Complex, FftHandler, R2cFftHandler};
// // Use parallel transforms:
// use ndrustfft::{
//     ndfft_par as ndfft, ndfft_r2c_par as ndfft_r2c, ndifft_par as ndifft,
//     ndifft_r2c_par as ndifft_r2c,
// };

fn main() {
    let (nx, ny) = (3, 3);

    // Init arrays
    let v = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],];
    let mut vhat: Array2<Complex<f64>> = Array2::zeros((nx, ny / 2 + 1));

    // Init handlers
    let mut handler_ax0 = FftHandler::<f64>::new(nx);
    let mut handler_ax1 = R2cFftHandler::<f64>::new(ny);

    // Transform
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((nx, ny / 2 + 1));
        ndfft_r2c(&v, &mut work, &mut handler_ax1, 1);
        ndfft(&work, &mut vhat, &mut handler_ax0, 0);
    }

    // Assert with numpys solution (rounded to five digits)
    let numpy_vhat = array![
        [Complex::new(45., 0.), Complex::new(-4.5, 2.59808)],
        [Complex::new(-13.5, 7.79423), Complex::new(0., 0.)],
        [Complex::new(-13.5, -7.79423), Complex::new(0., 0.)],
    ];
    Zip::from(&vhat).and(&numpy_vhat).for_each(|&x, &y| {
        if (x.re - y.re).abs() > 1e-4 || (x.im - y.im).abs() > 1e-4 {
            panic!("Large difference of values, got {} expected {}.", x, y)
        };
    });

    // Inverse Transform
    let mut v_new: Array2<f64> = Array2::zeros((nx, ny));
    {
        let mut work: Array2<Complex<f64>> = Array2::zeros((nx, ny / 2 + 1));
        ndifft(&vhat, &mut work, &mut handler_ax0, 0);
        ndifft_r2c(&work, &mut v_new, &mut handler_ax1, 1);
    }

    // Assert with original array
    Zip::from(&v_new).and(&v).for_each(|&x, &y| {
        if (x - y).abs() > 1e-4 {
            panic!("Large difference of values, got {} expected {}.", x, y)
        };
    });
}

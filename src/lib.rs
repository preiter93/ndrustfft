//! # ndRustfft: *n*-dimensional real-to-complex FFT and real-to-real DCT
//!
//! This library is a wrapper for RustFFT that enables performing FFT of real-valued data.
//! The library uses the *n*-dimensional Arrays of the ndarray library.
//!
//! ndRustfft provides Handler structs for FFT's and DCTs, which must be provided
//! to the respective ndfft, nddct function alongside with the correct Arrays.
//! The Handlers contain the transform plans and buffers to reduce allocation cost.
//! The Handlers implement a process function, which is a wrapper around Rustfft's
//! process function with additional steps, i.e. to provide a real-to complex fft,
//! or to construct the discrete cosine transform (dct) from a classical fft.
//!
//! The transform along the outermost axis are the cheapest, while transforms along
//! other axis' need to copy data temporary.
//!
//! ## Example
//! 2-Dimensional real-to-complex fft along first axis
//! ```
//! use ndarray::{Array, Dim, Ix};
//! use ndrustfft::{ndfft, Complex, FftHandler};
//!
//! let (nx, ny) = (6, 4);
//! let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
//! let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx, ny));
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f64;
//! }
//! let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
//! ndfft(
//!     &mut data.view_mut(),
//!     &mut vhat.view_mut(),
//!     &mut fft_handler,
//!     0,
//! );
//! ```

extern crate ndarray;
extern crate rustfft;
use ndarray::{Array1, ArrayViewMut, Dimension, RemoveAxis, Zip};
pub use rustfft::num_complex::Complex;
pub use rustfft::num_traits::Zero;
pub use rustfft::FftNum;
use rustfft::FftPlanner;
use std::sync::Arc;

/// Declare procedural macro which creates common function,
/// that defines how to iterate over the specified axis in a array.
/// The fft/dct transform is applied iteratively for each vector-lane.
macro_rules! create_transform {
    ($i: ident, $a: ty, $b: ty, $h: ty) => {
        pub fn $i<T, D>(
            input: &mut ArrayViewMut<$a, D>,
            output: &mut ArrayViewMut<$b, D>,
            handler: &mut $h,
            axis: usize,
        ) where
            T: FftNum,
            D: Dimension + RemoveAxis,
        {
            let outer_axis = input.ndim() - 1;
            if axis == outer_axis {
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .for_each(|x, mut y| {
                        handler.process(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                    });
            } else {
                let mut outvec = Array1::zeros(output.shape()[axis]);
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .for_each(|x, mut y| {
                        handler.process(&x.to_vec(), outvec.as_slice_mut().unwrap());
                        y.assign(&outvec);
                    });
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
            }
        }
    };
}

/// # *n*-dimensional real-to-complex Fourier Transform.
///
/// Performs best on sizes which are multiples of 2 or 3.
///
/// # Example
/// 2-Dimensional real-to-complex fft along first axis
/// ```
/// use ndarray::{Array, Dim, Ix};
/// use ndrustfft::{ndfft, Complex, FftHandler};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
/// ndfft(
///     &mut data.view_mut(),
///     &mut vhat.view_mut(),
///     &mut fft_handler,
///     0,
/// );
/// ```
///
/// The accompanying function is [`ndfft`].
pub struct FftHandler<T: FftNum> {
    pub n: usize,
    pub plan_fwd: Arc<dyn rustfft::Fft<T>>,
    pub buffer: Vec<Complex<T>>,
}

impl<T: FftNum> FftHandler<T> {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(n);
        let buffer = vec![Complex::zero(); n];
        FftHandler::<T> {
            n,
            plan_fwd: Arc::clone(&fft),
            buffer,
        }
    }

    pub fn process(&mut self, data: &[T], out: &mut [Complex<T>]) {
        self.assert_size(data.len());
        for (b, d) in self.buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::from_f64(0.0).unwrap());
        }
        self.plan_fwd.process(&mut self.buffer);
        for (b, d) in self.buffer[0..self.n / 2 + 1].iter().zip(out.iter_mut()) {
            *d = *b;
        }
    }

    fn assert_size(&self, size: usize) {
        assert!(
            self.n == size,
            "Size mismatch in fft, got {} expected {}",
            size,
            self.n
        );
    }
}

create_transform!(ndfft, T, Complex<T>, FftHandler<T>);

/// # *n*-dimensional real-to-real Cosine Transform (DCT-I).
///
/// Performs best on sizes where *2(n-1)* is a mutiple of 2 or 3.
///
/// # Example
/// 2-Dimensional real-to-real dft along second axis
/// ```
/// use ndarray::{Array, Dim, Ix};
/// use ndrustfft::{Dct1Handler, nddct1};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut handler: Dct1Handler<f64> = Dct1Handler::new(ny);
/// nddct1(
///     &mut data.view_mut(),
///     &mut vhat.view_mut(),
///     &mut handler,
///     1,
/// );
/// ```
///
/// The accompanying function is [`nddct1`].
pub struct Dct1Handler<T: FftNum> {
    pub n: usize,
    pub plan: Arc<dyn rustfft::Fft<T>>,
    pub buffer: Vec<Complex<T>>,
}

impl<T: FftNum> Dct1Handler<T> {
    pub fn new(n: usize) -> Self {
        let m = 2 * (n - 1);
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(m);
        let buffer = vec![Complex::zero(); m];
        Dct1Handler::<T> {
            n,
            plan: Arc::clone(&fft),
            buffer,
        }
    }

    /// # Perform 1-dimensional DCT-I on a vector.
    ///
    /// # Algorithm:
    /// 1. Reorder:
    /// [a,b,c,d] -> [a,b,c,d,c,b]
    ///
    /// 2. Compute FFT
    /// -> [a*,b*,c*,d*,c*,b*]
    ///
    /// 3. Extract
    /// [a*,b*,c*,d*]
    /// ```
    pub fn process(&mut self, data: &[T], out: &mut [T]) {
        self.assert_size(data.len());
        let n = self.buffer.len();
        self.buffer[0] = Complex::new(data[0], T::from_f64(0.0).unwrap());
        for (i, d) in data[1..].iter().enumerate() {
            self.buffer[i + 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
            self.buffer[n - i - 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
        }
        self.plan.process(&mut self.buffer);
        out[0] = self.buffer[0].re;
        for (i, d) in out[1..].iter_mut().enumerate() {
            *d = self.buffer[i + 1].re;
        }
    }

    fn assert_size(&self, size: usize) {
        assert!(
            self.n == size,
            "Size mismatch in dct, got {} expected {}",
            size,
            self.n
        );
    }
}

create_transform!(nddct1, T, T, Dct1Handler<T>);

// pub fn ndfft<T, D>(
//     input: &mut ArrayViewMut<T, D>,
//     output: &mut ArrayViewMut<Complex<T>, D>,
//     fft_handler: &mut FftHandler<T>,
//     axis: usize,
// ) where
//     T: FftNum,
//     D: Dimension + RemoveAxis,
// {
//     let outer_axis = input.ndim() - 1;
//     if axis == outer_axis {
//         Zip::from(input.rows())
//             .and(output.rows_mut())
//             .for_each(|x, mut y| {
//                 fft_handler.process(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
//             });
//     } else {
//         let mut outvec = Array1::zeros(output.shape()[axis]);
//         input.swap_axes(outer_axis, axis);
//         output.swap_axes(outer_axis, axis);
//         Zip::from(input.rows())
//             .and(output.rows_mut())
//             .for_each(|x, mut y| {
//                 fft_handler.process(&x.to_vec(), outvec.as_slice_mut().unwrap());
//                 y.assign(&outvec);
//             });
//         input.swap_axes(outer_axis, axis);
//         output.swap_axes(outer_axis, axis);
//     }
// }

// pub struct MyArray<T, D>(pub Array<T, D>);
//
// impl<T, D> MyArray<T, D>
// where
//     T: FftNum,
//     D: Dimension + RemoveAxis,
// {
//     pub fn fft(&self, output: &mut ArrayViewMut<Complex<T>, D>, fft_handler: &mut Rustfft<T>) {
//         let axis = self.outer_axis();
//         Zip::from(self.0.rows())
//             .and(output.rows_mut())
//             .for_each(|x, mut y| {
//                 println!("{:?} {:?}", x, y);
//                 fft_handler.fft(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
//                 println!("{:?} {:?}", x, y);
//                 println!("");
//             });
//     }
//
//     fn outer_axis(&self) -> usize {
//         self.0.ndim() - 1
//     }
// }

// pub fn rustfft1d<T: FftNum + std::ops::MulAssign>(rfft: &Rustfft<T>, data: &mut [Complex<T>]) {
//     rfft.fft.process(data);
// }
//
// pub fn rustfft1d_r2c<T>(rfft: &mut Rustfft<T>, data: &mut [T])
// where
//     T: FftNum,
//     f64: Into<T>,
// {
//     for (b, d) in rfft.buffer.iter_mut().zip(data.iter()) {
//         *b = Complex::new(*d, T::from_f64(0.0).unwrap());
//     }
//     rfft.fft.process(&mut rfft.buffer);
// }
//
// pub struct Rustfft<T: FftNum> {
//     pub n: usize,
//     pub fft: Arc<dyn rustfft::Fft<T>>,
//     pub fft2n: Arc<dyn rustfft::Fft<T>>,
//     pub buffer: Vec<Complex<T>>,
//     pub buffer2n: Vec<Complex<T>>,
// }
//
// impl<T: FftNum> Rustfft<T> {
//     pub fn new(n: usize) -> Self {
//         let mut planner = FftPlanner::<T>::new();
//         let fft = planner.plan_fft_forward(n);
//         let buffer = vec![Complex::zero(); n];
//         let n2 = 2 * (n - 1);
//         let fft2n = planner.plan_fft_forward(n2);
//         let buffer2n = vec![Complex::zero(); n2];
//         Rustfft::<T> {
//             n,
//             fft: Arc::clone(&fft),
//             fft2n: Arc::clone(&fft2n),
//             buffer,
//             buffer2n,
//         }
//     }
// }
//
// impl<T> Rustfft<T>
// where
//     T: FftNum,
//     f64: Into<T>,
// {
//     pub fn r2c(&mut self, data: &[T], out: &mut [Complex<T>]) {
//         self._r2c(data);
//         for (b, d) in self.buffer[0..self.n / 2 + 1].iter().zip(out.iter_mut()) {
//             *d = *b;
//         }
//     }
//
//     pub fn _r2c(&mut self, data: &[T]) {
//         for (b, d) in self.buffer.iter_mut().zip(data.iter()) {
//             *b = Complex::new(*d, T::from_f64(0.0).unwrap());
//         }
//         self.fft.process(&mut self.buffer);
//     }
//
//     /// Perform type 1 discrete cosine transform (DCT-I) with rustfft
//     ///
//     /// # Algorithm:
//     /// ```
//     /// [a,b,c,d] -> [a,b,c,d,c,b]
//     /// FFT on 2*(n-1) array -> [a*,b*,c*,d*,c*,b*]
//     /// Extract [a*,b*,c*,d*]
//     /// ```
//     pub fn dct(&mut self, data: &mut [T], out: &mut [T]) {
//         let n = self.buffer2n.len();
//         self.buffer2n[0] = Complex::new(data[0], T::from_f64(0.0).unwrap());
//         for (i, d) in data[1..].iter().enumerate() {
//             self.buffer2n[i + 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
//             self.buffer2n[n - i - 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
//         }
//         self.fft2n.process(&mut self.buffer2n);
//         out[0] = self.buffer2n[0].re;
//         for (i, d) in out[1..].iter_mut().enumerate() {
//             *d = self.buffer2n[i + 1].re;
//         }
//     }
// }

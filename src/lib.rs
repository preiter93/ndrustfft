//! # ndrustfft: *n*-dimensional real-to-complex FFT and real-to-real DCT
//!
//! This library is a wrapper for RustFFT that enables performing FFTs of real-valued data
//! and DCT's on *n*-dimensional arrays (ndarray).
//!
//! ndrustfft provides Handler structs for FFT's and DCTs, which must be provided
//! to the respective ndrfft, nddct function alongside with ArrayViews.
//! The Handlers contain the transform plans and buffers which reduce allocation cost.
//! The Handlers implement a process function, which is a wrapper around Rustfft's
//! process function with additional steps, i.e. to provide a real-to complex fft,
//! or to construct the discrete cosine transform (dct) from a classical fft.
//!
//! The transform along the outermost axis are the cheapest, while transforms along
//! other axis' need to copy data temporary.
//!
//! ## Parallel
//! The library ships all functions with a parallel version
//! which leverages the parallel abilities of ndarray.
//!
//! ## Example
//! 2-Dimensional real-to-complex fft along first axis
//! ```
//! use ndarray::{Array, Dim, Ix};
//! use ndrustfft::{ndrfft, Complex, FftHandler};
//!
//! let (nx, ny) = (6, 4);
//! let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
//! let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx / 2 + 1, ny));
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f64;
//! }
//! let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
//! ndrfft(
//!     &mut data.view_mut(),
//!     &mut vhat.view_mut(),
//!     &mut fft_handler,
//!     0,
//! );
//! ```
#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]
extern crate ndarray;
extern crate rustfft;
use ndarray::{Array1, ArrayViewMut, Dimension, RemoveAxis, Zip};
pub use rustfft::num_complex::Complex;
pub use rustfft::num_traits::Zero;
pub use rustfft::FftNum;
use rustfft::FftPlanner;
use std::sync::Arc;

/// Declare procedural macro which creates functions for the individual
/// transforms, i.e. fft, ifft and dct.
/// The fft/dct transforms are applied for each vector-lane along the
/// specified axis.
macro_rules! create_transform {
    (
        $(#[$meta:meta])* $i: ident, $a: ty, $b: ty, $h: ty, $p: ident
    ) => {
        $(#[$meta])*
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
                        handler.$p(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                    });
            } else {
                let mut outvec = Array1::zeros(output.shape()[axis]);
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .for_each(|x, mut y| {
                        handler.$p(&x.to_vec(), outvec.as_slice_mut().unwrap());
                        y.assign(&outvec);
                    });
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
            }
        }
    };
}

/// Similar to create_transform, but supports parallel computation.
macro_rules! create_transform_par {
    ($(#[$meta:meta])* $i: ident, $a: ty, $b: ty, $h: ty, $p: ident) => {
        $(#[$meta])*
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
                    .par_for_each(|x, mut y| {
                        handler.$p(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                    });
            } else {
                let n = output.shape()[axis];
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .par_for_each(|x, mut y| {
                        let mut outvec = Array1::zeros(n);
                        handler.$p(&x.to_vec(), outvec.as_slice_mut().unwrap());
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
/// Transforms a real ndarray of size *n* to a complex array of size
/// *n/2+1* and vice versa. The transformation is performed along a single
/// axis, all other array dimensions are unaffected.
/// Performs best on sizes which are mutiple of 2 or 3.
///
/// The accompanying functions for the forward transform are [`ndrfft`] (serial) and
/// [`ndrfft_par`] (parallel).
///
/// The accompanying functions for the inverse transform are [`ndirfft`] (serial) and
/// [`ndirfft_par`] (parallel).
///
/// # Example
/// 2-Dimensional real-to-complex fft along first axis
/// ```
/// use ndarray::{Array, Dim, Ix};
/// use ndrustfft::{ndrfft, Complex, FftHandler};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx / 2 + 1, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
/// ndrfft(
///     &mut data.view_mut(),
///     &mut vhat.view_mut(),
///     &mut fft_handler,
///     0,
/// );
/// ```
pub struct FftHandler<T: FftNum> {
    n: usize,
    m: usize,
    plan_fwd: Arc<dyn rustfft::Fft<T>>,
    plan_bwd: Arc<dyn rustfft::Fft<T>>,
    buffer: Vec<Complex<T>>,
}

impl<T: FftNum> FftHandler<T> {
    /// Creates a new FftHandler.
    ///
    /// # Arguments
    ///
    /// * `n` - Length of array along axis of which fft will be performed.
    /// The size of the complex array after the fft is performed will be of
    /// size *n / 2 + 1*.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndrustfft::FftHandler;
    /// let handler: FftHandler<f64> = FftHandler::new(10);
    /// ```
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<T>::new();
        let fwd = planner.plan_fft_forward(n);
        let bwd = planner.plan_fft_inverse(n);
        let buffer = vec![Complex::zero(); n];
        FftHandler::<T> {
            n,
            m: n / 2 + 1,
            plan_fwd: Arc::clone(&fwd),
            plan_bwd: Arc::clone(&bwd),
            buffer,
        }
    }

    fn rfft_lane(&mut self, data: &[T], out: &mut [Complex<T>]) {
        self.assert_size(self.n, data.len());
        self.assert_size(self.m, out.len());
        for (b, d) in self.buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::from_f64(0.0).unwrap());
        }
        self.plan_fwd.process(&mut self.buffer);
        for (b, d) in self.buffer[0..self.n / 2 + 1].iter().zip(out.iter_mut()) {
            *d = *b;
        }
    }

    fn rfft_lane_par(&self, data: &[T], out: &mut [Complex<T>]) {
        self.assert_size(self.n, data.len());
        self.assert_size(self.m, out.len());
        let mut buffer = vec![Complex::zero(); self.n];
        for (b, d) in buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::from_f64(0.0).unwrap());
        }
        self.plan_fwd.process(&mut buffer);
        for (b, d) in buffer[0..self.n / 2 + 1].iter().zip(out.iter_mut()) {
            *d = *b;
        }
    }

    fn irfft_lane(&mut self, data: &[Complex<T>], out: &mut [T]) {
        self.assert_size(self.m, data.len());
        self.assert_size(self.n, out.len());
        let m = data.len();
        for (b, d) in self.buffer[..m].iter_mut().zip(data.iter()) {
            *b = *d;
        }
        for (b, d) in self.buffer[m..].iter_mut().zip(data[1..].iter()) {
            b.re = d.re;
            b.im = -d.im;
        }
        self.plan_bwd.process(&mut self.buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in self.buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    fn irfft_lane_par(&self, data: &[Complex<T>], out: &mut [T]) {
        self.assert_size(self.m, data.len());
        let m = data.len();
        let mut buffer = vec![Complex::zero(); self.n];
        for (b, d) in buffer[..m].iter_mut().zip(data.iter()) {
            *b = *d;
        }
        for (b, d) in buffer[m..].iter_mut().zip(data[1..].iter()) {
            b.re = d.re;
            b.im = -d.im;
        }
        self.plan_bwd.process(&mut buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    fn assert_size(&self, n: usize, size: usize) {
        assert!(
            n == size,
            "Size mismatch in fft, got {} expected {}",
            size,
            n
        );
    }
}

create_transform!(
    /// Real-to-complex Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::{Array, Dim, Ix};
    /// use ndrustfft::{ndrfft, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndrfft(
    ///     &mut data.view_mut(),
    ///     &mut vhat.view_mut(),
    ///     &mut handler,
    ///     0,
    /// );
    ndrfft, T, Complex<T>, FftHandler<T>, rfft_lane);

create_transform!(
    /// Complex-to-real inverse Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::{Array, Dim, Ix};
    /// use ndrustfft::{ndirfft, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in vhat.iter_mut().enumerate() {
    ///     v.re = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndirfft(
    ///     &mut vhat.view_mut(),
    ///     &mut data.view_mut(),
    ///     &mut handler,
    ///     0,
    /// );
    ndirfft, Complex<T>, T, FftHandler<T>, irfft_lane);

create_transform_par!(
    /// Real-to-complex Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndrfft`]
    ndrfft_par, T, Complex<T>, FftHandler<T>, rfft_lane_par);
create_transform_par!(
    /// Complex-to-real inverse Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndirfft`]
    ndirfft_par, Complex<T>, T, FftHandler<T>, irfft_lane_par);

/// # *n*-dimensional real-to-real Cosine Transform (DCT-I).
///
/// The dct transforms a real ndarray of size *n* to a real array of size *n*.
/// The transformation is performed along a single axis, all other array
/// dimensions are unaffected.
/// Performs best on sizes where *2(n-1)* is a mutiple of 2 or 3. The crate
/// contains benchmarks, see benches folder, where different sizes can be
/// tested to optmize performance.
///
/// The accompanying functions are [`nddct1`] (serial) and
/// [`nddct1_par`] (parallel).
///
/// # Example
/// 2-Dimensional real-to-real dft along second axis
/// ```
/// use ndarray::{Array, Dim, Ix};
/// use ndrustfft::{DctHandler, nddct1};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut handler: DctHandler<f64> = DctHandler::new(ny);
/// nddct1(
///     &mut data.view_mut(),
///     &mut vhat.view_mut(),
///     &mut handler,
///     1,
/// );
/// ```
pub struct DctHandler<T: FftNum> {
    n: usize,
    plan: Arc<dyn rustfft::Fft<T>>,
    buffer: Vec<Complex<T>>,
}

impl<T: FftNum> DctHandler<T> {
    /// Creates a new DctHandler.
    ///
    /// # Arguments
    ///
    /// * `n` - Length of array along axis of which dct will be performed.
    /// The size and type of the array will be the same after the transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndrustfft::DctHandler;
    /// let handler: DctHandler<f64> = DctHandler::new(10);
    /// ```
    pub fn new(n: usize) -> Self {
        let m = 2 * (n - 1);
        let mut planner = FftPlanner::<T>::new();
        let fft = planner.plan_fft_forward(m);
        let buffer = vec![Complex::zero(); m];
        DctHandler::<T> {
            n,
            plan: Arc::clone(&fft),
            buffer,
        }
    }

    /// # Algorithm:
    /// 1. Reorder:
    /// (a,b,c,d) -> (a,b,c,d,c,b)
    ///
    /// 2. Compute FFT
    /// -> (a*,b*,c*,d*,c*,b*)
    ///
    /// 3. Extract
    /// (a*,b*,c*,d*)
    fn dct1_lane(&mut self, data: &[T], out: &mut [T]) {
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

    fn dct1_lane_par(&self, data: &[T], out: &mut [T]) {
        self.assert_size(data.len());
        let n = self.n;
        let mut buffer = vec![Complex::zero(); 2 * (self.n - 1)];
        buffer[0] = Complex::new(data[0], T::from_f64(0.0).unwrap());
        for (i, d) in data[1..].iter().enumerate() {
            buffer[i + 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
            buffer[n - i - 1] = Complex::new(*d, T::from_f64(0.0).unwrap());
        }
        self.plan.process(&mut buffer);
        out[0] = buffer[0].re;
        for (i, d) in out[1..].iter_mut().enumerate() {
            *d = buffer[i + 1].re;
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

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 1 DCT-I (serial).
    ///
    /// # Example
    /// ```
    /// use ndarray::{Array, Dim, Ix};
    /// use ndrustfft::{DctHandler, nddct1};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: DctHandler<f64> = DctHandler::new(ny);
    /// nddct1(
    ///     &mut data.view_mut(),
    ///     &mut vhat.view_mut(),
    ///     &mut handler,
    ///     1,
    /// );
    /// ```
    nddct1, T, T, DctHandler<T>, dct1_lane);
create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 1 DCT-I  (parallel).
    ///
    /// Further infos: see [`nddct1`]
    nddct1_par, T, T, DctHandler<T>, dct1_lane_par);

/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array, Dim, Ix};

    #[test]
    /// Successive forward and inverse transform
    fn test_fft() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
        let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx, ny / 2 + 1));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let mut handler: FftHandler<f64> = FftHandler::new(ny);
        let expected = data.clone();
        ndrfft(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 1);
        ndirfft(&mut vhat.view_mut(), &mut data.view_mut(), &mut handler, 1);

        // Assert
        let dif = 1e-6;
        for (a, b) in expected.iter().zip(data.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }
}

//! # ndrustfft: *n*-dimensional complex-to-complex FFT, real-to-complex FFT and real-to-real DCT
//!
//! This library is a wrapper for `RustFFT` that enables performing FFTs of complex-, real-valued
//! data and DCT's on *n*-dimensional arrays (ndarray).
//!
//! ndrustfft provides Handler structs for FFT's and DCTs, which must be provided
//! to the respective function (see implemented transforms below) alongside with the arrays.
//! The Handlers contain the transform plans and buffers which reduce allocation cost.
//! The Handlers implement a process function, which is a wrapper around Rustfft's
//! process function with additional functionality.
//! Transforms along the outermost axis are in general the fastest, while transforms along
//! other axis' will create temporary copies of the input array.
//!
//! ## Implemented transforms
//! ### Complex-to-complex
//! - `fft` / `ifft`: [`ndfft`],[`ndfft_par`], [`ndifft`],[`ndifft_par`]
//! ### Real-to-complex
//! - `fft_r2c` / `ifft_r2c`: [`ndfft_r2c`],[`ndfft_r2c_par`], [`ndifft_r2c`],[`ndifft_r2c_par`]
//! ### Real-to-real
//! - `fft_r2hc` / `ifft_r2hc`: [`ndfft_r2hc`],[`ndfft_r2hc_par`], [`ndifft_r2hc`],[`ndifft_r2hc_par`]
//! - `dct1`: [`nddct1`],[`nddct1_par`]
//!
//! ## Parallel
//! The library ships all functions with a parallel version
//! which leverages the parallel abilities of ndarray.
//!
//! ## Example
//! 2-Dimensional real-to-complex fft along first axis
//! ```
//! use ndarray::{Array2, Dim, Ix};
//! use ndrustfft::{ndfft_r2c, Complex, FftHandler};
//!
//! let (nx, ny) = (6, 4);
//! let mut data = Array2::<f64>::zeros((nx, ny));
//! let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f64;
//! }
//! let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
//! ndfft_r2c(
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
use ndarray::{Array1, ArrayBase, Dimension, Zip};
use ndarray::{Data, DataMut};
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
        pub fn $i<R, S, T, D>(
            input: &mut ArrayBase<R, D>,
            output: &mut ArrayBase<S, D>,
            handler: &mut $h,
            axis: usize,
        ) where
            T: FftNum,
            R: Data<Elem = $a>,
            S: Data<Elem = $b> + DataMut,
            D: Dimension,
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
        pub fn $i<R, S, T, D>(
            input: &mut ArrayBase<R, D>,
            output: &mut ArrayBase<S, D>,
            handler: &mut $h,
            axis: usize,
        ) where
            T: FftNum,
            R: Data<Elem = $a>,
            S: Data<Elem = $b> + DataMut,
            D: Dimension,
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
/// The accompanying functions for the forward transform are [`ndfft_r2c`] (serial) and
/// [`ndfft_r2c_par`] (parallel).
///
/// The accompanying functions for the inverse transform are [`ndifft_r2c`] (serial) and
/// [`ndifft_r2c_par`] (parallel).
///
/// # Example
/// 2-Dimensional real-to-complex fft along first axis
/// ```
/// use ndarray::{Array2, Dim, Ix};
/// use ndrustfft::{ndfft_r2c, Complex, FftHandler};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array2::<f64>::zeros((nx, ny));
/// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
/// ndfft_r2c(&mut data, &mut vhat, &mut fft_handler, 0);
/// ```
#[derive(Clone)]
pub struct FftHandler<T> {
    n: usize,
    m: usize,
    plan_fwd: Arc<dyn rustfft::Fft<T>>,
    plan_bwd: Arc<dyn rustfft::Fft<T>>,
    buffer: Vec<Complex<T>>,
}

impl<T: FftNum> FftHandler<T> {
    /// Creates a new `FftHandler`.
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
    #[allow(clippy::similar_names)]
    #[must_use]
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

    fn fft_lane(&self, data: &[Complex<T>], out: &mut [Complex<T>]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d;
        }
        self.plan_fwd.process(out);
    }

    #[allow(clippy::cast_precision_loss)]
    fn ifft_lane(&self, data: &[Complex<T>], out: &mut [Complex<T>]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d;
        }
        self.plan_bwd.process(out);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for b in out.iter_mut() {
            *b = *b * n64;
        }
    }

    fn fft_r2c_lane(&mut self, data: &[T], out: &mut [Complex<T>]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.m, out.len());
        for (b, d) in self.buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::zero());
        }
        self.plan_fwd.process(&mut self.buffer);
        for (b, d) in self.buffer[0..=self.n / 2].iter().zip(out.iter_mut()) {
            *d = *b;
        }
    }

    fn fft_r2c_lane_par(&self, data: &[T], out: &mut [Complex<T>]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.m, out.len());
        let mut buffer = vec![Complex::zero(); self.n];
        for (b, d) in buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::zero());
        }
        self.plan_fwd.process(&mut buffer);
        for (b, d) in buffer[0..=self.n / 2].iter().zip(out.iter_mut()) {
            *d = *b;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn ifft_r2c_lane(&mut self, data: &[Complex<T>], out: &mut [T]) {
        Self::assert_size(self.m, data.len());
        Self::assert_size(self.n, out.len());
        let m = data.len();
        for (b, d) in self.buffer[..m].iter_mut().zip(data.iter()) {
            *b = *d;
        }
        for (b, d) in self.buffer[m..].iter_mut().rev().zip(data[1..].iter()) {
            b.re = d.re;
            b.im = -d.im;
        }
        self.plan_bwd.process(&mut self.buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in self.buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn ifft_r2c_lane_par(&self, data: &[Complex<T>], out: &mut [T]) {
        Self::assert_size(self.m, data.len());
        let m = data.len();
        let mut buffer = vec![Complex::zero(); self.n];
        for (b, d) in buffer[..m].iter_mut().zip(data.iter()) {
            *b = *d;
        }
        for (b, d) in buffer[m..].iter_mut().rev().zip(data[1..].iter()) {
            b.re = d.re;
            b.im = -d.im;
        }
        self.plan_bwd.process(&mut buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    /// Real to half-complex [r0, r1, r2, r3, i2, i1]
    #[allow(clippy::cast_precision_loss)]
    fn fft_r2hc_lane(&mut self, data: &[T], out: &mut [T]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        for (b, d) in self.buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::zero());
        }
        self.plan_fwd.process(&mut self.buffer);
        // Transfer to half-complex format
        out[0] = self.buffer[0].re;
        out[self.n / 2] = self.buffer[self.n / 2].re;
        let (left, right) = out.split_at_mut(self.n / 2);
        for (b, (d1, d2)) in self.buffer[1..self.n / 2]
            .iter()
            .zip(left[1..].iter_mut().zip(right[1..].iter_mut().rev()))
        {
            *d1 = b.re;
            *d2 = b.im;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn fft_r2hc_lane_par(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        let mut buffer = vec![Complex::zero(); self.n];
        for (b, d) in buffer.iter_mut().zip(data.iter()) {
            *b = Complex::new(*d, T::zero());
        }
        self.plan_fwd.process(&mut buffer);
        // Transfer to half-complex format
        out[0] = buffer[0].re;
        out[self.n / 2] = buffer[self.n / 2].re;
        let (left, right) = out.split_at_mut(self.n / 2);
        for (b, (d1, d2)) in buffer[1..self.n / 2]
            .iter()
            .zip(left[1..].iter_mut().zip(right[1..].iter_mut().rev()))
        {
            *d1 = b.re;
            *d2 = b.im;
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::shadow_unrelated)]
    fn ifft_r2hc_lane(&mut self, data: &[T], out: &mut [T]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        self.buffer[0].re = data[0];
        self.buffer[0].im = T::zero();
        self.buffer[self.n / 2].re = data[self.n / 2];
        self.buffer[self.n / 2].im = T::zero();
        let (left, right) = data.split_at(self.n / 2);
        for (b, (d1, d2)) in self.buffer[1..self.n / 2]
            .iter_mut()
            .zip(left[1..].iter().zip(right[1..].iter().rev()))
        {
            b.re = *d1;
            b.im = *d2;
        }
        // Conjugate part
        let (left, right) = self.buffer.split_at_mut(self.n / 2);
        for (r, l) in right[1..].iter_mut().rev().zip(left[1..].iter()) {
            r.re = l.re;
            r.im = -l.im;
        }

        self.plan_bwd.process(&mut self.buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in self.buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::shadow_unrelated)]
    fn ifft_r2hc_lane_par(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.n, out.len());
        let mut buffer = vec![Complex::zero(); self.n];
        buffer[0].re = data[0];
        buffer[self.n / 2].re = data[self.n / 2];
        let (left, right) = data.split_at(self.n / 2);
        for (b, (d1, d2)) in buffer[1..self.n / 2]
            .iter_mut()
            .zip(left[1..].iter().zip(right[1..].iter().rev()))
        {
            b.re = *d1;
            b.im = *d2;
        }
        // Conjugate part
        let (left, right) = buffer.split_at_mut(self.n / 2);
        for (r, l) in right[1..].iter_mut().rev().zip(left[1..].iter()) {
            r.re = l.re;
            r.im = -l.im;
        }

        self.plan_bwd.process(&mut buffer);
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        for (b, d) in buffer.iter().zip(out.iter_mut()) {
            *d = b.re * n64;
        }
    }

    fn assert_size(n: usize, size: usize) {
        assert!(
            n == size,
            "Size mismatch in fft, got {} expected {}",
            size,
            n
        );
    }
}

create_transform!(
    /// Complex-to-complex Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::{Array2, Dim, Ix};
    /// use ndrustfft::{ndfft, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<Complex<f64>>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     v.re = i as f64;
    ///     v.im = -1.0*i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(ny);
    /// ndfft(&mut data, &mut vhat, &mut handler, 1);
    /// ```
    ndfft,
    Complex<T>,
    Complex<T>,
    FftHandler<T>,
    fft_lane
);

create_transform!(
    /// Complex-to-complex Inverse Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndfft, ndifft, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<Complex<f64>>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     v.re = i as f64;
    ///     v.im = -1.0*i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(ny);
    /// ndfft(&mut data, &mut vhat, &mut handler, 1);
    /// ndifft(&mut vhat, &mut data, &mut handler, 1);
    /// ```
    ndifft,
    Complex<T>,
    Complex<T>,
    FftHandler<T>,
    ifft_lane
);

create_transform!(
    /// Real-to-complex Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndfft_r2c, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndfft_r2c(&mut data, &mut vhat, &mut handler, 0);
    /// ```
    ndfft_r2c,
    T,
    Complex<T>,
    FftHandler<T>,
    fft_r2c_lane
);

create_transform!(
    /// Complex-to-real inverse Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndifft_r2c, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in vhat.iter_mut().enumerate() {
    ///     v.re = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndifft_r2c(&mut vhat, &mut data, &mut handler, 0);
    /// ```
    ndifft_r2c,
    Complex<T>,
    T,
    FftHandler<T>,
    ifft_r2c_lane
);

create_transform!(
    /// Real-to-real Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndfft_r2hc, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<f64>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndfft_r2hc(&mut data, &mut vhat, &mut handler, 0);
    /// ```
    ndfft_r2hc,
    T,
    T,
    FftHandler<T>,
    fft_r2hc_lane
);

create_transform!(
    /// Real-to-real Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndifft_r2hc, Complex, FftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<f64>::zeros((nx, ny));
    /// for (i, v) in vhat.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: FftHandler<f64> = FftHandler::new(nx);
    /// ndifft_r2hc(&mut vhat, &mut data, &mut handler, 0);
    /// ```
    ndifft_r2hc,
    T,
    T,
    FftHandler<T>,
    ifft_r2hc_lane
);

create_transform_par!(
    /// Complex-to-complex Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndfft`]
    ndfft_par,
    Complex<T>,
    Complex<T>,
    FftHandler<T>,
    fft_lane
);

create_transform_par!(
    /// Complex-to-complex inverse Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndifft`]
    ndifft_par,
    Complex<T>,
    Complex<T>,
    FftHandler<T>,
    ifft_lane
);

create_transform_par!(
    /// Real-to-complex Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndfft_r2c`]
    ndfft_r2c_par,
    T,
    Complex<T>,
    FftHandler<T>,
    fft_r2c_lane_par
);

create_transform_par!(
    /// Complex-to-real inverse Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndifft_r2c`]
    ndifft_r2c_par,
    Complex<T>,
    T,
    FftHandler<T>,
    ifft_r2c_lane_par
);

create_transform_par!(
    /// Real-to-real Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndfft_r2hc`]
    ndfft_r2hc_par,
    T,
    T,
    FftHandler<T>,
    fft_r2hc_lane_par
);

create_transform_par!(
    /// Real-to-real inverse Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndifft_r2hc`]
    ndifft_r2hc_par,
    T,
    T,
    FftHandler<T>,
    ifft_r2hc_lane_par
);

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
/// use ndarray::Array2;
/// use ndrustfft::{DctHandler, nddct1};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array2::<f64>::zeros((nx, ny));
/// let mut vhat = Array2::<f64>::zeros((nx, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut handler: DctHandler<f64> = DctHandler::new(ny);
/// nddct1(&mut data, &mut vhat, &mut handler, 1);
/// ```
#[derive(Clone)]
pub struct DctHandler<T> {
    n: usize,
    plan: Arc<dyn rustfft::Fft<T>>,
    buffer: Vec<Complex<T>>,
}

impl<T: FftNum> DctHandler<T> {
    /// Creates a new `DctHandler`.
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
    #[must_use]
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
        let m = self.buffer.len();
        for b in &mut self.buffer.iter_mut() {
            b.re = T::zero();
            b.im = T::zero();
        }
        self.buffer[0] = Complex::new(data[0], T::zero());
        for (i, d) in data[1..].iter().enumerate() {
            self.buffer[i + 1] = Complex::new(*d, T::zero());
            self.buffer[m - i - 1] = Complex::new(*d, T::zero());
        }
        self.plan.process(&mut self.buffer);
        out[0] = self.buffer[0].re;
        for (i, d) in out[1..].iter_mut().enumerate() {
            *d = self.buffer[i + 1].re;
        }
    }

    fn dct1_lane_par(&self, data: &[T], out: &mut [T]) {
        self.assert_size(data.len());
        let m = 2 * (self.n - 1);
        let mut buffer = vec![Complex::zero(); m];
        buffer[0] = Complex::new(data[0], T::zero());
        for (i, d) in data[1..].iter().enumerate() {
            buffer[i + 1] = Complex::new(*d, T::zero());
            buffer[m - i - 1] = Complex::new(*d, T::zero());
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
    /// use ndarray::Array2;
    /// use ndrustfft::{DctHandler, nddct1};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<f64>::zeros((nx, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler: DctHandler<f64> = DctHandler::new(ny);
    /// nddct1(&mut data, &mut vhat, &mut handler, 1);
    /// ```
    nddct1,
    T,
    T,
    DctHandler<T>,
    dct1_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 1 DCT-I  (parallel).
    ///
    /// Further infos: see [`nddct1`]
    nddct1_par,
    T,
    T,
    DctHandler<T>,
    dct1_lane_par
);

/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Array2};

    fn approx_eq<A, S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        A: FftNum + std::fmt::Display + std::cmp::PartialOrd,
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        let dif = A::from_f64(1e-3).unwrap();
        for (a, b) in expected.iter().zip(result.iter()) {
            if (*a - *b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn approx_eq_complex<A, S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        A: FftNum + std::fmt::Display + std::cmp::PartialOrd,
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        let dif = A::from_f64(1e-3).unwrap();
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a.re - b.re).abs() > dif || (a.im - b.im).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    fn test_matrix() -> Array2<f64> {
        array![
            [0.1, 1.908, -0.035, -0.278, 0.264, -1.349],
            [0.88, 0.86, -0.267, -0.809, 1.374, 0.757],
            [1.418, -0.68, 0.814, 0.852, -0.613, 0.468],
            [0.817, -0.697, -2.157, 0.447, -0.949, 2.243],
            [-0.474, -0.09, -0.567, -0.772, 0.021, 2.455],
            [-0.745, 1.52, 0.509, -0.066, 2.802, -0.042],
        ]
    }

    fn test_matrix_complex() -> Array2<Complex<f64>> {
        test_matrix().mapv(|x| Complex::new(x, x))
    }

    #[test]
    fn test_fft() {
        // Solution from np.fft.fft
        let solution_re = array![
            [0.61, 3.105, 2.508, 0.048, -3.652, -2.019],
            [2.795, 0.612, 0.219, 1.179, -2.801, 3.276],
            [2.259, 0.601, 0.045, 0.979, 4.506, 0.118],
            [-0.296, -0.896, 0.544, -4.282, 3.544, 6.288],
            [0.573, -0.96, -3.85, -2.613, -0.461, 4.467],
            [3.978, -2.229, 0.133, 1.154, -6.544, -0.962],
        ];

        let solution_im = array![
            [0.61, -2.019, -3.652, 0.048, 2.508, 3.105],
            [2.795, 3.276, -2.801, 1.179, 0.219, 0.612],
            [2.259, 0.118, 4.506, 0.979, 0.045, 0.601],
            [-0.296, 6.288, 3.544, -4.282, 0.544, -0.896],
            [0.573, 4.467, -0.461, -2.613, -3.85, -0.96],
            [3.978, -0.962, -6.544, 1.154, 0.133, -2.229],
        ];

        let mut solution: Array2<Complex<f64>> = Array2::zeros(solution_re.raw_dim());
        for (s, (s_re, s_im)) in solution
            .iter_mut()
            .zip(solution_re.iter().zip(solution_im.iter()))
        {
            s.re = *s_re;
            s.im = *s_im;
        }

        // Setup
        let mut v = test_matrix_complex();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft(&mut v, &mut vhat, &mut handler, 1);
        ndifft(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq_complex(&v, &v_copy);
    }

    #[test]
    fn test_fft_r2c() {
        // Solution from np.fft.rfft
        let solution_re = array![
            [0.61, 0.543, -0.572, 0.048],
            [2.795, 1.944, -1.291, 1.179],
            [2.259, 0.36, 2.275, 0.979],
            [-0.296, 2.696, 2.044, -4.282],
            [0.573, 1.753, -2.155, -2.613],
            [3.978, -1.596, -3.205, 1.154],
        ];

        let solution_im = array![
            [0., -2.562, -3.08, 0.],
            [0., 1.332, -1.51, 0.],
            [0., -0.242, 2.23, 0.],
            [0., 3.592, 1.5, 0.],
            [0., 2.713, 1.695, 0.],
            [0., 0.633, -3.339, 0.],
        ];

        let mut solution: Array2<Complex<f64>> = Array2::zeros(solution_re.raw_dim());
        for (s, (s_re, s_im)) in solution
            .iter_mut()
            .zip(solution_re.iter().zip(solution_im.iter()))
        {
            s.re = *s_re;
            s.im = *s_im;
        }

        // Setup
        let mut v = test_matrix();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny / 2 + 1));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft_r2c(&mut v, &mut vhat, &mut handler, 1);
        ndifft_r2c(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq(&v, &v_copy);
    }

    #[test]
    fn test_fft_r2hc() {
        // Solution from np.fft.rfft
        let solution = array![
            [0.61, 0.543, -0.572, 0.048, -3.08, -2.562],
            [2.795, 1.944, -1.291, 1.179, -1.51, 1.332],
            [2.259, 0.36, 2.275, 0.979, 2.23, -0.242],
            [-0.296, 2.696, 2.044, -4.282, 1.5, 3.592],
            [0.573, 1.753, -2.155, -2.613, 1.695, 2.713],
            [3.978, -1.596, -3.205, 1.154, -3.339, 0.633],
        ];

        // Setup
        let mut v = test_matrix();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft_r2hc(&mut v, &mut vhat, &mut handler, 1);
        ndifft_r2hc(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
        approx_eq(&v, &v_copy);
    }

    #[test]
    fn test_fft_dct1() {
        // Solution from scipy.fft.dct(x, type=1)
        let solution = array![
            [2.469, 4.259, 0.6, 0.04, -4.957, -1.353],
            [3.953, -0.374, 4.759, -0.436, -2.643, 2.235],
            [2.632, 0.818, -1.609, 1.053, 5.008, 1.008],
            [-3.652, -2.628, 4.81, 2.632, 4.666, -7.138],
            [-0.835, -2.982, 4.105, -3.192, 1.265, -2.297],
            [8.743, -2.422, 1.167, -0.841, -7.506, 3.011],
        ];

        // Setup
        let mut v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct1(&mut v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }

    #[test]
    fn test_fft_par() {
        // Solution from np.fft.fft
        let solution_re = array![
            [0.61, 3.105, 2.508, 0.048, -3.652, -2.019],
            [2.795, 0.612, 0.219, 1.179, -2.801, 3.276],
            [2.259, 0.601, 0.045, 0.979, 4.506, 0.118],
            [-0.296, -0.896, 0.544, -4.282, 3.544, 6.288],
            [0.573, -0.96, -3.85, -2.613, -0.461, 4.467],
            [3.978, -2.229, 0.133, 1.154, -6.544, -0.962],
        ];

        let solution_im = array![
            [0.61, -2.019, -3.652, 0.048, 2.508, 3.105],
            [2.795, 3.276, -2.801, 1.179, 0.219, 0.612],
            [2.259, 0.118, 4.506, 0.979, 0.045, 0.601],
            [-0.296, 6.288, 3.544, -4.282, 0.544, -0.896],
            [0.573, 4.467, -0.461, -2.613, -3.85, -0.96],
            [3.978, -0.962, -6.544, 1.154, 0.133, -2.229],
        ];

        let mut solution: Array2<Complex<f64>> = Array2::zeros(solution_re.raw_dim());
        for (s, (s_re, s_im)) in solution
            .iter_mut()
            .zip(solution_re.iter().zip(solution_im.iter()))
        {
            s.re = *s_re;
            s.im = *s_im;
        }

        // Setup
        let mut v = test_matrix_complex();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft_par(&mut v, &mut vhat, &mut handler, 1);
        ndifft_par(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq_complex(&v, &v_copy);
    }

    #[test]
    fn test_fft_r2c_par() {
        // Solution from np.fft.rfft
        let solution_re = array![
            [0.61, 0.543, -0.572, 0.048],
            [2.795, 1.944, -1.291, 1.179],
            [2.259, 0.36, 2.275, 0.979],
            [-0.296, 2.696, 2.044, -4.282],
            [0.573, 1.753, -2.155, -2.613],
            [3.978, -1.596, -3.205, 1.154],
        ];

        let solution_im = array![
            [0., -2.562, -3.08, 0.],
            [0., 1.332, -1.51, 0.],
            [0., -0.242, 2.23, 0.],
            [0., 3.592, 1.5, 0.],
            [0., 2.713, 1.695, 0.],
            [0., 0.633, -3.339, 0.],
        ];

        let mut solution: Array2<Complex<f64>> = Array2::zeros(solution_re.raw_dim());
        for (s, (s_re, s_im)) in solution
            .iter_mut()
            .zip(solution_re.iter().zip(solution_im.iter()))
        {
            s.re = *s_re;
            s.im = *s_im;
        }

        // Setup
        let mut v = test_matrix();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny / 2 + 1));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft_r2c_par(&mut v, &mut vhat, &mut handler, 1);
        ndifft_r2c_par(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq(&v, &v_copy);
    }

    #[test]
    fn test_fft_r2hc_par() {
        // Solution from np.fft.rfft
        let solution = array![
            [0.61, 0.543, -0.572, 0.048, -3.08, -2.562],
            [2.795, 1.944, -1.291, 1.179, -1.51, 1.332],
            [2.259, 0.36, 2.275, 0.979, 2.23, -0.242],
            [-0.296, 2.696, 2.044, -4.282, 1.5, 3.592],
            [0.573, 1.753, -2.155, -2.613, 1.695, 2.713],
            [3.978, -1.596, -3.205, 1.154, -3.339, 0.633],
        ];

        // Setup
        let mut v = test_matrix();
        let v_copy = v.clone();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        // Transform
        ndfft_r2hc_par(&mut v, &mut vhat, &mut handler, 1);
        ndifft_r2hc_par(&mut vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
        approx_eq(&v, &v_copy);
    }

    #[test]
    fn test_fft_dct1_par() {
        // Solution from scipy.fft.dct(x, type=1)
        let solution = array![
            [2.469, 4.259, 0.6, 0.04, -4.957, -1.353],
            [3.953, -0.374, 4.759, -0.436, -2.643, 2.235],
            [2.632, 0.818, -1.609, 1.053, 5.008, 1.008],
            [-3.652, -2.628, 4.81, 2.632, 4.666, -7.138],
            [-0.835, -2.982, 4.105, -3.192, 1.265, -2.297],
            [8.743, -2.422, 1.167, -0.841, -7.506, 3.011],
        ];

        // Setup
        let mut v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct1_par(&mut v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }
}

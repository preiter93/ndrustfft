//! # ndrustfft: *n*-dimensional complex-to-complex FFT, real-to-complex FFT and real-to-real DCT
//!
//! This library is a wrapper for `RustFFT`, `RustDCT` and `RealFft`
//! that enables performing FFTs and DCTs of complex- and real-valued
//! data on *n*-dimensional arrays (ndarray).
//!
//! ndrustfft provides Handler structs for FFT's and DCTs, which must be provided alongside
//! with the arrays to the respective functions (see below) .
//! The Handlers implement a process function, which is a wrapper around Rustfft's
//! process.
//! Transforms along the outermost axis are in general the fastest, while transforms along
//! other axis' will temporarily create copies of the input array.
//!
//! ## Parallel
//! The library ships all functions with a parallel version
//! which leverages the parallel iterators of the ndarray crate.
//!
//! ## Available transforms
//! ### Complex-to-complex
//! - `fft` : [`ndfft`], [`ndfft_par`]
//! - `ifft`: [`ndifft`],[`ndifft_par`]
//! ### Real-to-complex
//! - `fft_r2c` : [`ndfft_r2c`], [`ndfft_r2c_par`],
//! ### Complex-to-real
//! - `ifft_r2c`: [`ndifft_r2c`],[`ndifft_r2c_par`]
//! ### Real-to-real
//! *Discrete Cosine Transform*
//! - `dct1`: [`nddct1`],[`nddct1_par`]
//! - `dct2`: [`nddct2`],[`nddct2_par`]
//! - `dct3`: [`nddct3`],[`nddct3_par`]
//! - `dct4`: [`nddct4`],[`nddct4_par`]
//!
//! ## Example
//! 2-Dimensional real-to-complex fft along first axis
//! ```
//! use ndarray::{Array2, Dim, Ix};
//! use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
//!
//! let (nx, ny) = (6, 4);
//! let mut data = Array2::<f64>::zeros((nx, ny));
//! let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
//! for (i, v) in data.iter_mut().enumerate() {
//!     *v = i as f64;
//! }
//! let mut fft_handler = R2cFftHandler::<f64>::new(nx);
//! ndfft_r2c(
//!     &data.view(),
//!     &mut vhat.view_mut(),
//!     &mut fft_handler,
//!     0,
//! );
//! ```
//!
//! # Versions
//! - v0.3.0: Upgrade `RealFft` to 3.0.0 and `RustDCT` to 0.7
//! - \>= v0.2.2:
//!
//! The first and last elements of real-to-complex transforms are
//! per definition purely real. This is now enforced actively, by
//! setting the complex part to zero - similar to numpys rfft.
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
extern crate ndarray;
extern crate rustfft;
use ndarray::{Array1, ArrayBase, Dimension, Zip};
use ndarray::{Data, DataMut};
use num_traits::FloatConst;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustdct::{Dct1, DctPlanner, TransformType2And3, TransformType4};
pub use rustfft::num_complex::Complex;
pub use rustfft::num_traits::Zero;
pub use rustfft::FftNum;
use rustfft::{Fft, FftPlanner};
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
            input: &ArrayBase<R, D>,
            output: &mut ArrayBase<S, D>,
            handler: &mut $h,
            axis: usize,
        ) where
            T: FftNum + FloatConst,
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
                let mut input = input.view();
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .for_each(|x, mut y| {
                        handler.$p(&x.to_vec(), outvec.as_slice_mut().unwrap());
                        y.assign(&outvec);
                    });
                output.swap_axes(outer_axis, axis);
            }
        }
    };
}

/// Similar to ``create_transform``, but supports parallel computation.
macro_rules! create_transform_par {
    ($(#[$meta:meta])* $i: ident, $a: ty, $b: ty, $h: ty, $p: ident) => {
        $(#[$meta])*
        pub fn $i<R, S, T, D>(
            input: &ArrayBase<R, D>,
            output: &mut ArrayBase<S, D>,
            handler: &mut $h,
            axis: usize,
        ) where
            T: FftNum + FloatConst,
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
                let mut input = input.view();
                input.swap_axes(outer_axis, axis);
                output.swap_axes(outer_axis, axis);
                Zip::from(input.rows())
                    .and(output.rows_mut())
                    .par_for_each(|x, mut y| {
                        let mut outvec = Array1::zeros(n);
                        handler.$p(&x.to_vec(), outvec.as_slice_mut().unwrap());
                        y.assign(&outvec);
                    });
                output.swap_axes(outer_axis, axis);
            }
        }
    };
}

/// # *n*-dimensional complex-to-complex Fourier Transform.
///
/// Transforms a complex ndarray of size *n* to a complex array of size
/// *n* and vice versa. The transformation is performed along a single
/// axis, all other array dimensions are unaffected.
/// Performs best on sizes which are mutiple of 2 or 3.
///
/// The accompanying functions for the forward transform are [`ndfft`] (serial) and
/// [`ndfft_par`] (parallel).
///
/// The accompanying functions for the inverse transform are [`ndifft`] (serial) and
/// [`ndifft_par`] (parallel).
///
/// # Example
/// 2-Dimensional complex-to-complex fft along first axis
/// ```
/// use ndarray::{Array2, Dim, Ix};
/// use ndrustfft::{ndfft, Complex, FftHandler};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array2::<Complex<f64>>::zeros((nx, ny));
/// let mut vhat = Array2::<Complex<f64>>::zeros((nx, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     v.re = i as f64;
///     v.im = i as f64;
/// }
/// let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
/// ndfft(&data, &mut vhat, &mut fft_handler, 0);
/// ```
#[derive(Clone)]
pub struct FftHandler<T> {
    n: usize,
    plan_fwd: Arc<dyn Fft<T>>,
    plan_bwd: Arc<dyn Fft<T>>,
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
        FftHandler::<T> {
            n,
            plan_fwd: Arc::clone(&fwd),
            plan_bwd: Arc::clone(&bwd),
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
    /// ndfft(&data, &mut vhat, &mut handler, 1);
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
    /// ndfft(&data, &mut vhat, &mut handler, 1);
    /// ndifft(&vhat, &mut data, &mut handler, 1);
    /// ```
    ndifft,
    Complex<T>,
    Complex<T>,
    FftHandler<T>,
    ifft_lane
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

// create_transform_par!(
//     /// Real-to-complex Fourier Transform (parallel).
//     ///
//     /// Further infos: see [`ndfft_r2c`]
//     ndfft_r2c_par,
//     T,
//     Complex<T>,
//     FftHandler<T>,
//     fft_r2c_lane_par
// );

// create_transform_par!(
//     /// Complex-to-real inverse Fourier Transform (parallel).
//     ///
//     /// Further infos: see [`ndifft_r2c`]
//     ndifft_r2c_par,
//     Complex<T>,
//     T,
//     FftHandler<T>,
//     ifft_r2c_lane_par
// );

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
/// use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
///
/// let (nx, ny) = (6, 4);
/// let mut data = Array2::<f64>::zeros((nx, ny));
/// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
/// for (i, v) in data.iter_mut().enumerate() {
///     *v = i as f64;
/// }
/// let mut fft_handler = R2cFftHandler::<f64>::new(nx);
/// ndfft_r2c(&data, &mut vhat, &mut fft_handler, 0);
/// ```
#[derive(Clone)]
pub struct R2cFftHandler<T> {
    n: usize,
    m: usize,
    plan_fwd: Arc<dyn RealToComplex<T>>,
    plan_bwd: Arc<dyn ComplexToReal<T>>,
}

impl<T: FftNum> R2cFftHandler<T> {
    /// Creates a new `RealFftPlanner`.
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
    /// use ndrustfft::R2cFftHandler;
    /// let handler = R2cFftHandler::<f64>::new(10);
    /// ```
    #[allow(clippy::similar_names)]
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<T>::new();
        let fwd = planner.plan_fft_forward(n);
        let bwd = planner.plan_fft_inverse(n);
        Self {
            n,
            m: n / 2 + 1,
            plan_fwd: Arc::clone(&fwd),
            plan_bwd: Arc::clone(&bwd),
        }
    }

    fn fft_r2c_lane(&self, data: &[T], out: &mut [Complex<T>]) {
        Self::assert_size(self.n, data.len());
        Self::assert_size(self.m, out.len());
        let mut indata = vec![T::zero(); self.n];
        for (a, b) in indata.iter_mut().zip(data.iter()) {
            *a = *b;
        }
        self.plan_fwd.process(&mut indata, out).unwrap();
    }

    #[allow(clippy::cast_precision_loss)]
    fn ifft_r2c_lane(&self, data: &[Complex<T>], out: &mut [T]) {
        Self::assert_size(self.m, data.len());
        Self::assert_size(self.n, out.len());
        let n64 = T::from_f64(1. / self.n as f64).unwrap();
        let mut indata = vec![Complex::zero(); self.m];
        for (a, b) in indata.iter_mut().zip(data.iter()) {
            a.re = b.re * n64;
            a.im = b.im * n64;
        }
        // First element must be real
        indata[0].im = T::zero();
        // If original vector is even, last element must be real
        if self.n % 2 == 0 {
            indata[self.m - 1].im = T::zero();
        }
        self.plan_bwd.process(&mut indata, out).unwrap();
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
    /// Real-to-complex Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in data.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let mut handler = R2cFftHandler::<f64>::new(nx);
    /// ndfft_r2c(&data, &mut vhat, &mut handler, 0);
    /// ```
    ndfft_r2c,
    T,
    Complex<T>,
    R2cFftHandler<T>,
    fft_r2c_lane
);

create_transform!(
    /// Complex-to-real inverse Fourier Transform (serial).
    /// # Example
    /// ```
    /// use ndarray::Array2;
    /// use ndrustfft::{ndifft_r2c, Complex, R2cFftHandler};
    ///
    /// let (nx, ny) = (6, 4);
    /// let mut data = Array2::<f64>::zeros((nx, ny));
    /// let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
    /// for (i, v) in vhat.iter_mut().enumerate() {
    ///     v.re = i as f64;
    /// }
    /// let mut handler = R2cFftHandler::<f64>::new(nx);
    /// ndifft_r2c(&vhat, &mut data, &mut handler, 0);
    /// ```
    ndifft_r2c,
    Complex<T>,
    T,
    R2cFftHandler<T>,
    ifft_r2c_lane
);

create_transform_par!(
    /// Real-to-complex Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndfft_r2c`]
    ndfft_r2c_par,
    T,
    Complex<T>,
    R2cFftHandler<T>,
    fft_r2c_lane
);

create_transform_par!(
    /// Complex-to-real inverse Fourier Transform (parallel).
    ///
    /// Further infos: see [`ndifft_r2c`]
    ndifft_r2c_par,
    Complex<T>,
    T,
    R2cFftHandler<T>,
    ifft_r2c_lane
);

/// # *n*-dimensional real-to-real Cosine Transform.
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
/// nddct1(&data, &mut vhat, &mut handler, 1);
/// ```
#[derive(Clone)]
pub struct DctHandler<T> {
    n: usize,
    plan_dct1: Arc<dyn Dct1<T>>,
    plan_dct2: Arc<dyn TransformType2And3<T>>,
    plan_dct3: Arc<dyn TransformType2And3<T>>,
    plan_dct4: Arc<dyn TransformType4<T>>,
}

impl<T: FftNum + FloatConst> DctHandler<T> {
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
        let mut planner = DctPlanner::<T>::new();
        let dct1 = planner.plan_dct1(n);
        let dct2 = planner.plan_dct2(n);
        let dct3 = planner.plan_dct3(n);
        let dct4 = planner.plan_dct4(n);
        Self {
            n,
            plan_dct1: Arc::clone(&dct1),
            plan_dct2: Arc::clone(&dct2),
            plan_dct3: Arc::clone(&dct3),
            plan_dct4: Arc::clone(&dct4),
        }
    }

    fn dct1_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct1.process_dct1(out);
    }

    fn dct1_ortho_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let n = out.len();
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        // Normalize
        out[0] = out[0] * T::SQRT_2();
        out[n - 1] = out[n - 1] * T::SQRT_2();
        self.plan_dct1.process_dct1(out);
        let f = T::one() / two * T::from_f64((2. / (n - 1) as f64).sqrt()).unwrap();
        for x in out[1..n - 1].iter_mut() {
            *x = *x * f;
        }
        out[0] = out[0] * f / T::SQRT_2();
        out[n - 1] = out[n - 1] * f / T::SQRT_2();
    }

    fn dct2_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct2.process_dct2(out);
    }

    fn dct2_ortho_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let n = out.len();
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct2.process_dct2(out);
        let f = T::from_f64((1. / (2 * n) as f64).sqrt()).unwrap();
        for x in out[1..].iter_mut() {
            *x = *x * f;
        }
        out[0] = out[0] * f / T::SQRT_2();
    }

    fn dct3_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct3.process_dct3(out);
    }

    fn dct3_ortho_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let n = out.len();
        let f = T::SQRT_2() / T::from_f64((n as f64).sqrt()).unwrap();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * f;
        }
        out[0] = out[0] * T::SQRT_2();
        self.plan_dct3.process_dct3(out);
    }

    fn dct4_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct4.process_dct4(out);
    }

    fn dct4_ortho_lane(&self, data: &[T], out: &mut [T]) {
        Self::assert_size(self, data.len());
        Self::assert_size(self, out.len());
        let n = out.len();
        let two = T::one() + T::one();
        for (b, d) in out.iter_mut().zip(data.iter()) {
            *b = *d * two;
        }
        self.plan_dct4.process_dct4(out);
        let f = T::from_f64((1. / (2 * n) as f64).sqrt()).unwrap();
        for x in out.iter_mut() {
            *x = *x * f;
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
    /// # Definition
    /// $
    /// y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
    /// \frac{\pi k n}{N-1} \right)
    /// $
    ///
    /// see also *Pythons* *scipy* library (``norm=None``). To replicate
    /// ``norm=None`` use [`nddct1_ortho`].
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
    /// nddct1(&data, &mut vhat, &mut handler, 1);
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
    dct1_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 2 DCT-2 (serial).
    ///
    /// # Definition
    /// $
    /// y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)
    /// $
    ///
    /// see also *Pythons* *scipy* library (``norm=None``). To replicate
    /// ``norm=None`` use [`nddct2_ortho`].
    nddct2,
    T,
    T,
    DctHandler<T>,
    dct2_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 2 DCT-2  (parallel).
    nddct2_par,
    T,
    T,
    DctHandler<T>,
    dct2_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 3 DCT-3 (serial).
    ///
    /// # Definition
    /// $
    /// y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)
    /// $
    ///
    /// see also *Pythons* *scipy* library (``norm=None``). To replicate
    /// ``norm=None`` use [`nddct3_ortho`].
    nddct3,
    T,
    T,
    DctHandler<T>,
    dct3_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 3 DCT-3  (parallel).
    nddct3_par,
    T,
    T,
    DctHandler<T>,
    dct3_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 4 DCT-4 (serial).
    ///
    /// # Definition
    /// $
    /// y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)
    /// $
    ///
    /// see also *Pythons* *scipy* library (``norm=None``). To replicate
    /// ``norm=None`` use [`nddct4_ortho`].
    nddct4,
    T,
    T,
    DctHandler<T>,
    dct4_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 4 DCT-4  (parallel).
    nddct4_par,
    T,
    T,
    DctHandler<T>,
    dct4_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 1 DCT-1 (serial).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    ///
    /// # Definition
    /// $
    /// y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
    /// \frac{\pi k n}{N-1} \right)
    /// $
    ///
    /// ``x[0]`` and ``x[N-1]`` are multiplied by a scaling
    /// factor of $\sqrt{2}$, and ``y[k]`` is multiplied by a scaling factor
    /// ``f``:
    ///
    /// $
    ///     f = \begin{cases}
    ///     \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
    ///     \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}
    /// $
    ///
    /// See also *Pythons* *scipy* library (``norm=ortho``).
    nddct1_ortho,
    T,
    T,
    DctHandler<T>,
    dct1_ortho_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 1 DCT-1  (parallel).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    nddct1_ortho_par,
    T,
    T,
    DctHandler<T>,
    dct1_ortho_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 2 DCT-2 (serial).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    ///
    /// # Definition
    /// $
    /// y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)
    /// $
    ///
    /// ``y[k]`` is multiplied by a scaling factor
    /// ``f``:
    ///
    /// $
    ///     f = \begin{cases}
    ///   \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
    ///   \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}
    /// $
    ///
    /// See also *Pythons* *scipy* library (``norm=ortho``).
    nddct2_ortho,
    T,
    T,
    DctHandler<T>,
    dct2_ortho_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 2 DCT-2  (parallel).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    nddct2_ortho_par,
    T,
    T,
    DctHandler<T>,
    dct2_ortho_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 3 DCT-3 (serial).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    ///
    /// # Definition
    /// $
    /// y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
    ///   \cos\left(\frac{\pi(2k+1)n}{2N}\right)
    /// $
    ///
    /// See also *Pythons* *scipy* library (``norm=ortho``).
    nddct3_ortho,
    T,
    T,
    DctHandler<T>,
    dct3_ortho_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 3 DCT-3  (parallel).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    nddct3_ortho_par,
    T,
    T,
    DctHandler<T>,
    dct3_ortho_lane
);

create_transform!(
    /// Real-to-real Discrete Cosine Transform of type 4 DCT-4 (serial).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    ///
    /// # Definition
    /// $
    /// y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)
    /// $
    ///
    /// ``y[k]`` is multiplied by a scaling factor
    /// ``f``:
    ///
    /// $
    ///     f = \frac{1}{\sqrt{2N}}
    /// $
    ///
    /// See also *Pythons* *scipy* library (``norm=ortho``).
    nddct4_ortho,
    T,
    T,
    DctHandler<T>,
    dct4_ortho_lane
);

create_transform_par!(
    /// Real-to-real Discrete Cosine Transform of type 4 DCT-4  (parallel).
    ///
    /// Normalization mode: "ortho", i.e. equal to *Matlabs* ``dct(x)``,
    /// see *scipys* documentation
    nddct4_ortho_par,
    T,
    T,
    DctHandler<T>,
    dct4_ortho_lane
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
        ndfft(&v, &mut vhat, &mut handler, 1);
        ndifft(&vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq_complex(&v, &v_copy);

        // Transform Par
        let mut v = test_matrix_complex();
        ndfft_par(&v, &mut vhat, &mut handler, 1);
        ndifft_par(&vhat, &mut v, &mut handler, 1);

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
        let mut handler = R2cFftHandler::<f64>::new(ny);

        // Transform
        ndfft_r2c(&v, &mut vhat, &mut handler, 1);
        ndifft_r2c(&vhat, &mut v, &mut handler, 1);

        // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq(&v, &v_copy);

        // Transform Par
        let mut v = test_matrix();
        ndfft_r2c_par(&v, &mut vhat, &mut handler, 1);
        ndifft_r2c_par(&vhat, &mut v, &mut handler, 1);

        // // Assert
        approx_eq_complex(&vhat, &solution);
        approx_eq(&v, &v_copy);
    }

    #[test]
    fn test_ifft_c2r_first_last_element() {
        let n = 6;
        let mut v = Array1::<f64>::zeros(n);
        let mut vhat = Array1::<Complex<f64>>::zeros(n / 2 + 1);
        let solution_numpy_first_elem: Array1<f64> =
            array![0.16667, 0.16667, 0.16667, 0.16667, 0.16667, 0.16667];
        let solution_numpy_last_elem: Array1<f64> =
            array![0.16667, -0.16667, 0.16667, -0.16667, 0.16667, -0.16667];
        let mut rfft_handler = R2cFftHandler::<f64>::new(n);

        // First element should be purely real, thus the imaginary
        // part should not matter. However, original realfft gives
        // different results for different imaginary parts
        vhat[0].re = 1.;
        vhat[0].im = 100.;
        // backward
        ndifft_r2c(&vhat, &mut v, &mut rfft_handler, 0);
        // assert
        approx_eq(&v, &solution_numpy_first_elem);

        // Same for last element, if input is even
        for v in vhat.iter_mut() {
            v.re = 0.;
            v.im = 0.;
        }
        vhat[3].re = 1.;
        vhat[3].im = 100.;
        // backward
        ndifft_r2c(&vhat, &mut v, &mut rfft_handler, 0);
        // assert
        approx_eq(&v, &solution_numpy_last_elem);
    }

    #[test]
    fn test_dct1_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct1(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![24., -6.82842712, 0., -1.17157288, 0.];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct1_ortho_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct1_ortho(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![6.62132034, -3., 0.87867966, -1., 0.62132034];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct2_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct2(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![30., -9.95959314, 0., -0.898055953, 0.];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct2_ortho_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct2_ortho(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![6.70820393, -3.14949989, 0., -0.28399023, 0.];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct3_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct3(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![17.45077999, -14.20158303, 5., -3.68696079, 0.43776383];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct3_ortho_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct3_ortho(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![5.649407, -4.35994905, 1.71212466, -1.03493354, 0.26941891];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct4_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct4(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![14.97831211, -14.2763015, 7.07106781, -6.4587212, 5.48837883];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct4_ortho_lane() {
        let mut handler: DctHandler<f64> = DctHandler::new(5);
        let x = array![1., 2., 3., 4., 5.];
        let mut y = Array1::zeros(5);
        nddct4_ortho(&x, &mut y, &mut handler, 0);
        // Assert
        let scipy_dct = vec![4.73655818, -4.51456293, 2.23606798, -2.04242698, 1.73557778];
        for (a, b) in y.iter().zip(scipy_dct.iter()) {
            assert!((*a - *b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dct1() {
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
        let v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct1(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);

        // Transform Par
        let v = test_matrix();
        nddct1_par(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }

    #[test]
    fn test_dct2() {
        // Solution from scipy.fft.dct(x, type=2)
        let solution = array![
            [1.22, 5.25, -1.621, -0.619, -5.906, -1.105],
            [5.59, -0.209, 4.699, 0.134, -3.907, 1.838],
            [4.518, 1.721, 0.381, 1.492, 6.138, 0.513],
            [-0.592, -3.746, 8.262, 1.31, 4.642, -6.125],
            [1.146, -5.709, 5.75, -4.275, 0.78, -0.963],
            [7.956, -2.873, -2.13, 0.006, -8.988, 2.56],
        ];

        // Setup
        let v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct2(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);

        // Transform Par
        let v = test_matrix();
        nddct2_par(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }

    #[test]
    fn test_dct3() {
        // Solution from scipy.fft.dct(x, type=3)
        let solution = array![
            [2.898, 4.571, -0.801, 1.65, -5.427, -2.291],
            [2.701, -0.578, 5.768, -0.335, -3.158, 0.882],
            [2.348, -0.184, -1.258, 0.048, 5.472, 2.081],
            [-3.421, -2.075, 6.944, 0.264, 7.505, -4.315],
            [-1.43, -3.023, 6.317, -5.259, 1.991, -1.44],
            [5.76, -4.047, 1.974, 0.376, -8.651, 0.117],
        ];

        // Setup
        let v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct3(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);

        // Transform Par
        let v = test_matrix();
        nddct3_par(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }

    #[test]
    fn test_dct4() {
        // Solution from scipy.fft.dct(x, type=4)
        let solution = array![
            [3.18, 2.73, -2.314, -2.007, -5.996, 2.127],
            [3.175, 0.865, 4.939, -4.305, -0.443, 1.568],
            [3.537, 0.677, 0.371, 4.186, 4.528, -1.531],
            [-2.687, 1.838, 6.968, 0.899, 2.456, -8.79],
            [-2.289, -1.002, 3.67, -5.705, 3.867, -4.349],
            [4.192, -5.626, 1.789, -6.057, -4.61, 4.627],
        ];

        // Setup
        let v = test_matrix();
        let (nx, ny) = (v.shape()[0], v.shape()[1]);
        let mut vhat = Array2::<f64>::zeros((nx, ny));
        let mut handler: DctHandler<f64> = DctHandler::new(ny);

        // Transform
        nddct4(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);

        // Transform Par
        let v = test_matrix();
        nddct4_par(&v, &mut vhat, &mut handler, 1);

        // Assert
        approx_eq(&vhat, &solution);
    }
}

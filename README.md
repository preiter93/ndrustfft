# ndrustfft

## ndrustfft: *n*-dimensional complex-to-complex FFT, real-to-complex FFT and real-to-real DCT

This library is a wrapper for `RustFFT`, `RustDCT` and `RealFft`
that enables performing FFTs and DCTs of complex- and real-valued
data on *n*-dimensional arrays (ndarray).

ndrustfft provides Handler structs for FFT's and DCTs, which must be provided alongside
with the arrays to the respective function (see below) .
The Handlers implement a process function, which is a wrapper around Rustfft's
process function with additional functionality.
Transforms along the outermost axis are in general the fastest, while transforms along
other axis' will create temporary copies of the input array.

### Implemented transforms
#### Complex-to-complex
- `fft` : [`ndfft`], [`ndfft_par`]
- `ifft`: [`ndifft`],[`ndifft_par`]
#### Real-to-complex
- `fft_r2c` : [`ndfft_r2c`], [`ndfft_r2c_par`],
#### Complex-to-real
- `ifft_r2c`: [`ndifft_r2c`],[`ndifft_r2c_par`]
#### Real-to-real
- `dct1`: [`nddct1`],[`nddct1_par`]
- `dct2`: [`nddct2`],[`nddct2_par`]
- `dct3`: [`nddct3`],[`nddct3_par`]
- `dct4`: [`nddct4`],[`nddct4_par`]

`ndrustfft` >= v0.2.2:

Real-to-complex transforms now behave like numpys rfft.
The first element (for odd and even input) and the last element (for even input)
of the coefficient array should be real due to Hermitian symmetry.
Thus, the solution of the inverse transform is independent of the imaginary
part of the first and last element (for even input). Note, this is different
to the behaviour of the `RealFft` crate.

### Parallel
The library ships all functions with a parallel version
which leverages the parallel abilities of ndarray.

### Example
2-Dimensional real-to-complex fft along first axis
```rust
use ndarray::{Array2, Dim, Ix};
use ndrustfft::{ndfft_r2c, Complex, R2cFftHandler};

let (nx, ny) = (6, 4);
let mut data = Array2::<f64>::zeros((nx, ny));
let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
for (i, v) in data.iter_mut().enumerate() {
    *v = i as f64;
}
let mut fft_handler = R2cFftHandler::<f64>::new(nx);
ndfft_r2c(
    &data.view(),
    &mut vhat.view_mut(),
    &mut fft_handler,
    0,
);
```

License: MIT

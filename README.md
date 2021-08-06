# ndrustfft

## ndrustfft: *n*-dimensional complex-to-complex FFT, real-to-complex FFT and real-to-real DCT

This library is a wrapper for `RustFFT` that enables performing FFTs of complex-, real-valued
data and DCT's on *n*-dimensional arrays (ndarray).

ndrustfft provides Handler structs for FFT's and DCTs, which must be provided
to the respective function (see implemented transforms below) alongside with the arrays.
The Handlers contain the transform plans and buffers which reduce allocation cost.
The Handlers implement a process function, which is a wrapper around Rustfft's
process function with additional functionality.
Transforms along the outermost axis are in general the fastest, while transforms along
other axis' will create temporary copies of the input array.

### Implemented transforms
#### Complex-to-complex
- `fft` / `ifft`: [`ndfft`],[`ndfft_par`], [`ndifft`],[`ndifft_par`]
#### Real-to-complex
- `fft_r2c` / `ifft_r2c`: [`ndfft_r2c`],[`ndfft_r2c_par`], [`ndifft_r2c`],[`ndifft_r2c_par`]
#### Real-to-real
- `fft_r2hc` / `ifft_r2hc`: [`ndfft_r2hc`],[`ndfft_r2hc_par`], [`ndifft_r2hc`],[`ndifft_r2hc_par`]
- `dct1`: [`nddct1`],[`nddct1_par`]

### Parallel
The library ships all functions with a parallel version
which leverages the parallel abilities of ndarray.

### Example
2-Dimensional real-to-complex fft along first axis
```rust
use ndarray::{Array2, Dim, Ix};
use ndrustfft::{ndfft_r2c, Complex, FftHandler};

let (nx, ny) = (6, 4);
let mut data = Array2::<f64>::zeros((nx, ny));
let mut vhat = Array2::<Complex<f64>>::zeros((nx / 2 + 1, ny));
for (i, v) in data.iter_mut().enumerate() {
    *v = i as f64;
}
let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
ndfft_r2c(
    &mut data.view_mut(),
    &mut vhat.view_mut(),
    &mut fft_handler,
    0,
);
```

License: MIT

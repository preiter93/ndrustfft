# ndrustfft

## ndrustfft: *n*-dimensional real-to-complex FFT and real-to-real DCT

This library is a wrapper for RustFFT that enables performing FFTs of real-valued data
and DCT's on *n*-dimensional arrays (ndarray).

ndrustfft provides Handler structs for FFT's and DCTs, which must be provided
to the respective ndrfft, nddct function alongside with ArrayViews.
The Handlers contain the transform plans and buffers which reduce allocation cost.
The Handlers implement a process function, which is a wrapper around Rustfft's
process function with additional steps, i.e. to provide a real-to complex fft,
or to construct the discrete cosine transform (dct) from a classical fft.

The transform along the outermost axis are the cheapest, while transforms along
other axis' need to copy data temporary.

### Parallel
The library ships all functions with a parallel version
which leverages the parallel abilities of ndarray.

### Example
2-Dimensional real-to-complex fft along first axis
```rust
use ndarray::{Array, Dim, Ix};
use ndrustfft::{ndrfft, Complex, FftHandler};

let (nx, ny) = (6, 4);
let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx / 2 + 1, ny));
for (i, v) in data.iter_mut().enumerate() {
    *v = i as f64;
}
let mut fft_handler: FftHandler<f64> = FftHandler::new(nx);
ndrfft(
    &mut data.view_mut(),
    &mut vhat.view_mut(),
    &mut fft_handler,
    0,
);
```

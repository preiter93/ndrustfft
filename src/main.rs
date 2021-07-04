use ndarray::{Array, Dim, Ix};
//use ndrustfft::{nddct1, DctHandler};
use ndrustfft::{ndrfft, ndirfft, Complex, FftHandler};

fn main() {
    let (nx, ny) = (6, 4);
    let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((nx, ny / 2 + 1));
    for (i, v) in vhat.iter_mut().enumerate() {
        v.re = i as f64;
    }
    let mut handler: FftHandler<f64> = FftHandler::new(ny);
    ndirfft(&mut vhat.view_mut(), &mut data.view_mut(), &mut handler, 1);
    ndrfft(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 1);
}

use ndarray::{Array, Dim, Ix};
use ndrustfft::{nddct1, Dct1Handler};

fn main() {
    let (nx, ny) = (6, 4);
    let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f64;
    }
    let mut handler: Dct1Handler<f64> = Dct1Handler::new(ny);
    nddct1(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 1);
}

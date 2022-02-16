use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Dim, Ix};
use ndrustfft::{nddct1_par, DctHandler};
use ndrustfft::{ndfft_par, Complex, FftHandler};
use ndrustfft::{ndfft_r2c_par, R2cFftHandler};
const FFT_SIZES: [usize; 4] = [128, 264, 512, 1024];
const DCT_SIZES: [usize; 4] = [129, 265, 513, 1025];

pub fn bench_fft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d_par");
    for n in FFT_SIZES.iter() {
        let name = format!("Size: {}", *n);
        let mut data = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((*n, *n));
        let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((*n, *n));
        for (i, v) in data.iter_mut().enumerate() {
            v.re = i as f64;
            v.im = i as f64;
        }
        let mut handler: FftHandler<f64> = FftHandler::new(*n);
        group.bench_function(&name, |b| {
            b.iter(|| ndfft_par(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 0))
        });
    }
    group.finish();
}

pub fn bench_rfft2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft2d_par");
    for n in FFT_SIZES.iter() {
        let name = format!("Size: {}", *n);
        let m = *n / 2 + 1;
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((*n, *n));
        let mut vhat = Array::<Complex<f64>, Dim<[Ix; 2]>>::zeros((m, *n));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let mut handler = R2cFftHandler::<f64>::new(*n);
        group.bench_function(&name, |b| {
            b.iter(|| ndfft_r2c_par(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 0))
        });
    }
    group.finish();
}

pub fn bench_dct2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct2d_par");
    for n in DCT_SIZES.iter() {
        let name = format!("Size: {}", *n);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((*n, *n));
        let mut vhat = Array::<f64, Dim<[Ix; 2]>>::zeros((*n, *n));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let mut handler: DctHandler<f64> = DctHandler::new(*n);
        group.bench_function(&name, |b| {
            b.iter(|| nddct1_par(&mut data.view_mut(), &mut vhat.view_mut(), &mut handler, 0))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fft2d, bench_rfft2d, bench_dct2d);
criterion_main!(benches);

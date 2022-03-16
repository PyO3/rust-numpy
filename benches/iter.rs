#![feature(test)]
#![allow(deprecated)]

extern crate test;
use test::{black_box, Bencher};

use ndarray::Zip;
use numpy::{npyiter::NpyMultiIterBuilder, PyArray};
use pyo3::Python;

fn numpy_iter(bencher: &mut Bencher, size: usize) {
    Python::with_gil(|py| {
        let x = PyArray::<f64, _>::zeros(py, size, false);
        let y = PyArray::<f64, _>::zeros(py, size, false);
        let z = PyArray::<f64, _>::zeros(py, size, false);

        let x = x.readonly();
        let y = y.readonly();
        let mut z = z.readwrite();

        bencher.iter(|| {
            let iter = NpyMultiIterBuilder::new()
                .add_readonly(black_box(&x))
                .add_readonly(black_box(&y))
                .add_readwrite(black_box(&mut z))
                .build()
                .unwrap();

            for (x, y, z) in iter {
                *z = x + y;
            }
        });
    });
}

#[bench]
fn numpy_iter_small(bencher: &mut Bencher) {
    numpy_iter(bencher, 2_usize.pow(5));
}

#[bench]
fn numpy_iter_medium(bencher: &mut Bencher) {
    numpy_iter(bencher, 2_usize.pow(10));
}

#[bench]
fn numpy_iter_large(bencher: &mut Bencher) {
    numpy_iter(bencher, 2_usize.pow(15));
}

fn ndarray_iter(bencher: &mut Bencher, size: usize) {
    Python::with_gil(|py| {
        let x = PyArray::<f64, _>::zeros(py, size, false);
        let y = PyArray::<f64, _>::zeros(py, size, false);
        let z = PyArray::<f64, _>::zeros(py, size, false);

        let x = x.readonly();
        let y = y.readonly();
        let mut z = z.readwrite();

        bencher.iter(|| {
            Zip::from(black_box(x.as_array()))
                .and(black_box(y.as_array()))
                .and(black_box(z.as_array_mut()))
                .for_each(|x, y, z| {
                    *z = x + y;
                });
        });
    });
}

#[bench]
fn ndarray_iter_small(bencher: &mut Bencher) {
    ndarray_iter(bencher, 2_usize.pow(5));
}

#[bench]
fn ndarray_iter_medium(bencher: &mut Bencher) {
    ndarray_iter(bencher, 2_usize.pow(10));
}

#[bench]
fn ndarray_iter_large(bencher: &mut Bencher) {
    ndarray_iter(bencher, 2_usize.pow(15));
}

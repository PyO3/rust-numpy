#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use numpy::{PyArray, PyArrayMethods};
use pyo3::Python;

#[bench]
fn initial_shared_borrow(bencher: &mut Bencher) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        bencher.iter(|| {
            let array = black_box(&array);

            let _shared = array.readonly();
        });
    });
}

#[bench]
fn additional_shared_borrow(bencher: &mut Bencher) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        let _shared = (0..128).map(|_| array.readonly()).collect::<Vec<_>>();

        bencher.iter(|| {
            let array = black_box(&array);

            let _shared = array.readonly();
        });
    });
}

#[bench]
fn exclusive_borrow(bencher: &mut Bencher) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        bencher.iter(|| {
            let array = black_box(&array);

            let _exclusive = array.readwrite();
        });
    });
}

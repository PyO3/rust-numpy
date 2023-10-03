#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use std::ops::Range;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{PyAny, Python, ToPyObject};

#[bench]
fn extract_success(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        let any: &PyAny = PyArray2::<f64>::zeros(py, (10, 10), false);

        bencher.iter(|| {
            black_box(any).extract::<&PyArray2<f64>>().unwrap();
        });
    });
}

#[bench]
fn extract_failure(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        let any: &PyAny = PyArray2::<i32>::zeros(py, (10, 10), false);

        bencher.iter(|| {
            black_box(any).extract::<&PyArray2<f64>>().unwrap_err();
        });
    });
}

#[bench]
fn downcast_success(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        let any: &PyAny = PyArray2::<f64>::zeros(py, (10, 10), false);

        bencher.iter(|| {
            black_box(any).downcast::<PyArray2<f64>>().unwrap();
        });
    });
}

#[bench]
fn downcast_failure(bencher: &mut Bencher) {
    Python::with_gil(|py| {
        let any: &PyAny = PyArray2::<i32>::zeros(py, (10, 10), false);

        bencher.iter(|| {
            black_box(any).downcast::<PyArray2<f64>>().unwrap_err();
        });
    });
}

struct Iter(Range<usize>);

impl Iterator for Iter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

fn from_iter(bencher: &mut Bencher, size: usize) {
    iter_with_gil(bencher, |py| {
        let iter = black_box(Iter(0..size));

        PyArray1::from_iter(py, iter);
    });
}

#[bench]
fn from_iter_small(bencher: &mut Bencher) {
    from_iter(bencher, 2_usize.pow(5));
}

#[bench]
fn from_iter_medium(bencher: &mut Bencher) {
    from_iter(bencher, 2_usize.pow(10));
}

#[bench]
fn from_iter_large(bencher: &mut Bencher) {
    from_iter(bencher, 2_usize.pow(15));
}

fn from_slice(bencher: &mut Bencher, size: usize) {
    let vec = (0..size).collect::<Vec<_>>();

    iter_with_gil(bencher, |py| {
        let slice = black_box(&vec);

        PyArray1::from_slice(py, slice);
    });
}

#[bench]
fn from_slice_small(bencher: &mut Bencher) {
    from_slice(bencher, 2_usize.pow(5));
}

#[bench]
fn from_slice_medium(bencher: &mut Bencher) {
    from_slice(bencher, 2_usize.pow(10));
}

#[bench]
fn from_slice_large(bencher: &mut Bencher) {
    from_slice(bencher, 2_usize.pow(15));
}

fn from_object_slice(bencher: &mut Bencher, size: usize) {
    let vec = Python::with_gil(|py| (0..size).map(|val| val.to_object(py)).collect::<Vec<_>>());

    iter_with_gil(bencher, |py| {
        let slice = black_box(&vec);

        PyArray1::from_slice(py, slice);
    });
}

#[bench]
fn from_object_slice_small(bencher: &mut Bencher) {
    from_object_slice(bencher, 2_usize.pow(5));
}

#[bench]
fn from_object_slice_medium(bencher: &mut Bencher) {
    from_object_slice(bencher, 2_usize.pow(10));
}

#[bench]
fn from_object_slice_large(bencher: &mut Bencher) {
    from_object_slice(bencher, 2_usize.pow(15));
}

fn from_vec2(bencher: &mut Bencher, size: usize) {
    let vec2 = vec![vec![0; size]; size];

    iter_with_gil(bencher, |py| {
        let vec2 = black_box(&vec2);

        PyArray2::from_vec2(py, vec2).unwrap();
    });
}

#[bench]
fn from_vec2_small(bencher: &mut Bencher) {
    from_vec2(bencher, 2_usize.pow(3));
}

#[bench]
fn from_vec2_medium(bencher: &mut Bencher) {
    from_vec2(bencher, 2_usize.pow(5));
}

#[bench]
fn from_vec2_large(bencher: &mut Bencher) {
    from_vec2(bencher, 2_usize.pow(8));
}

fn from_vec3(bencher: &mut Bencher, size: usize) {
    let vec3 = vec![vec![vec![0; size]; size]; size];

    iter_with_gil(bencher, |py| {
        let vec3 = black_box(&vec3);

        PyArray3::from_vec3(py, vec3).unwrap();
    });
}

#[bench]
fn from_vec3_small(bencher: &mut Bencher) {
    from_vec3(bencher, 2_usize.pow(2));
}

#[bench]
fn from_vec3_medium(bencher: &mut Bencher) {
    from_vec3(bencher, 2_usize.pow(4));
}

#[bench]
fn from_vec3_large(bencher: &mut Bencher) {
    from_vec3(bencher, 2_usize.pow(5));
}

fn iter_with_gil(bencher: &mut Bencher, mut f: impl FnMut(Python<'_>)) {
    Python::with_gil(|py| {
        bencher.iter(|| {
            let pool = unsafe { py.new_pool() };

            f(pool.python());
        });
    });
}

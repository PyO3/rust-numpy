#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use std::ops::Range;

use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{types::PyAnyMethods, Bound, IntoPyObjectExt, Python};

#[bench]
fn extract_success(bencher: &mut Bencher) {
    Python::attach(|py| {
        let any = PyArray2::<f64>::zeros(py, (10, 10), false).into_any();

        bencher.iter(|| {
            black_box(&any)
                .extract::<Bound<'_, PyArray2<f64>>>()
                .unwrap()
        });
    });
}

#[bench]
fn extract_failure(bencher: &mut Bencher) {
    Python::attach(|py| {
        let any = PyArray2::<i32>::zeros(py, (10, 10), false).into_any();

        bencher.iter(|| {
            black_box(&any)
                .extract::<Bound<'_, PyArray2<f64>>>()
                .unwrap_err()
        });
    });
}

#[bench]
fn cast_success(bencher: &mut Bencher) {
    Python::attach(|py| {
        let any = PyArray2::<f64>::zeros(py, (10, 10), false).into_any();

        bencher.iter(|| black_box(&any).cast::<PyArray2<f64>>().unwrap());
    });
}

#[bench]
fn cast_failure(bencher: &mut Bencher) {
    Python::attach(|py| {
        let any = PyArray2::<i32>::zeros(py, (10, 10), false).into_any();

        bencher.iter(|| black_box(&any).cast::<PyArray2<f64>>().unwrap_err());
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
    Python::attach(|py| {
        bencher.iter(|| {
            let iter = black_box(Iter(0..size));

            PyArray1::from_iter(py, iter)
        });
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

    Python::attach(|py| {
        bencher.iter(|| {
            let slice = black_box(&vec);

            PyArray1::from_slice(py, slice)
        });
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
    let vec = Python::attach(|py| {
        (0..size)
            .map(|val| val.into_py_any(py).unwrap())
            .collect::<Vec<_>>()
    });

    Python::attach(|py| {
        bencher.iter(|| {
            let slice = black_box(&vec);

            PyArray1::from_slice(py, slice)
        });
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

    Python::attach(|py| {
        bencher.iter(|| {
            let vec2 = black_box(&vec2);

            PyArray2::from_vec2(py, vec2).unwrap()
        });
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

    Python::attach(|py| {
        bencher.iter(|| {
            let vec3 = black_box(&vec3);

            PyArray3::from_vec3(py, vec3).unwrap()
        });
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

#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use std::ops::Range;

use numpy::PyArray1;
use pyo3::{Python, ToPyObject};

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

struct ExactIter(Range<usize>);

impl Iterator for ExactIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for ExactIter {
    fn len(&self) -> usize {
        self.0.len()
    }
}

fn from_exact_iter(bencher: &mut Bencher, size: usize) {
    iter_with_gil(bencher, |py| {
        let iter = black_box(ExactIter(0..size));

        PyArray1::from_exact_iter(py, iter);
    });
}

#[bench]
fn from_exact_iter_small(bencher: &mut Bencher) {
    from_exact_iter(bencher, 2_usize.pow(5));
}

#[bench]
fn from_exact_iter_medium(bencher: &mut Bencher) {
    from_exact_iter(bencher, 2_usize.pow(10));
}

#[bench]
fn from_exact_iter_large(bencher: &mut Bencher) {
    from_exact_iter(bencher, 2_usize.pow(15));
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

fn iter_with_gil(bencher: &mut Bencher, mut f: impl FnMut(Python)) {
    Python::with_gil(|py| {
        bencher.iter(|| {
            let pool = unsafe { py.new_pool() };

            f(pool.python());
        });
    });
}

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use numpy::{PyArray, PyArrayMethods};
use pyo3::Python;

fn initial_shared_borrow(c: &mut Criterion) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        c.bench_function("initial_shared_borrow", |bencher| {
            bencher.iter(|| {
                let array = black_box(&array);

                let _shared = black_box(array.readonly());
            });
        });
    });
}

fn additional_shared_borrow(c: &mut Criterion) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        let _shared = (0..128).map(|_| array.readonly()).collect::<Vec<_>>();

        c.bench_function("additional_shared_borrow", |bencher| {
            bencher.iter(|| {
                let array = black_box(&array);

                let _shared = black_box(array.readonly());
            });
        });
    });
}

fn exclusive_borrow(c: &mut Criterion) {
    Python::attach(|py| {
        let array = PyArray::<f64, _>::zeros(py, (6, 5, 4, 3, 2, 1), false);

        c.bench_function("exclusive_borrow", |bencher| {
            bencher.iter(|| {
                let array = black_box(&array);

                let _exclusive = black_box(array.readwrite());
            });
        });
    });
}

criterion_group!(
    benches,
    initial_shared_borrow,
    additional_shared_borrow,
    exclusive_borrow
);
criterion_main!(benches);

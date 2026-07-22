use std::{hint::black_box, ops::Range};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{types::PyAnyMethods, Bound, IntoPyObjectExt, Python};

fn extract_success(c: &mut Criterion) {
    Python::attach(|py| {
        let any = PyArray2::<f64>::zeros(py, (10, 10), false).into_any();

        c.bench_function("extract_success", |b| {
            b.iter(|| {
                black_box(&any)
                    .extract::<Bound<'_, PyArray2<f64>>>()
                    .unwrap()
            });
        });
    });
}

fn extract_failure(c: &mut Criterion) {
    Python::attach(|py| {
        let any = PyArray2::<i32>::zeros(py, (10, 10), false).into_any();

        c.bench_function("extract_failure", |b| {
            b.iter(|| {
                black_box(&any)
                    .extract::<Bound<'_, PyArray2<f64>>>()
                    .unwrap_err()
            });
        });
    });
}

fn cast_success(c: &mut Criterion) {
    Python::attach(|py| {
        let any = PyArray2::<f64>::zeros(py, (10, 10), false).into_any();

        c.bench_function("cast_success", |b| {
            b.iter(|| black_box(&any).cast::<PyArray2<f64>>().unwrap());
        });
    });
}

fn cast_failure(c: &mut Criterion) {
    Python::attach(|py| {
        let any = PyArray2::<i32>::zeros(py, (10, 10), false).into_any();

        c.bench_function("cast_failure", |b| {
            b.iter(|| black_box(&any).cast::<PyArray2<f64>>().unwrap_err());
        })
    });
}

struct Iter(Range<usize>);

impl Iterator for Iter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

fn from_iter(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(5), 2_usize.pow(10), 2_usize.pow(15)];

    let mut group = c.benchmark_group("from_iter");
    for &size in SIZES {
        Python::attach(|py| {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let iter = black_box(Iter(0..size));
                    black_box(PyArray1::from_iter(py, iter));
                });
            });
        });
    }
}

fn from_slice(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(5), 2_usize.pow(10), 2_usize.pow(15)];

    let mut group = c.benchmark_group("from_slice");
    for &size in SIZES {
        let vec = (0..size).collect::<Vec<_>>();

        Python::attach(|py| {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let slice = black_box(&vec[..]);
                    black_box(PyArray1::from_slice(py, slice));
                });
            });
        });
    }
}

fn from_object_slice(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(5), 2_usize.pow(10), 2_usize.pow(15)];

    let mut group = c.benchmark_group("from_object_slice");
    for &size in SIZES {
        Python::attach(|py| {
            let vec = (0..size)
                .map(|val| val.into_py_any(py).unwrap())
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let slice = black_box(&vec[..]);
                    black_box(PyArray1::from_slice(py, slice));
                });
            });
        });
    }
}

fn from_vec2(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(3), 2_usize.pow(5), 2_usize.pow(8)];

    let mut group = c.benchmark_group("from_vec2");
    for &size in SIZES {
        let vec2 = vec![vec![0; size]; size];

        Python::attach(|py| {
            group.throughput(Throughput::Elements(size.pow(2) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let vec2 = black_box(&vec2);
                    black_box(PyArray2::from_vec2(py, vec2).unwrap());
                });
            });
        });
    }
}

fn from_vec3(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(2), 2_usize.pow(4), 2_usize.pow(5)];

    let mut group = c.benchmark_group("from_vec3");
    for &size in SIZES {
        let vec3 = vec![vec![vec![0; size]; size]; size];

        Python::attach(|py| {
            group.throughput(Throughput::Elements(size.pow(3) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let vec3 = black_box(&vec3);
                    black_box(PyArray3::from_vec3(py, vec3).unwrap());
                });
            });
        });
    }
}

criterion_group!(
    benches,
    extract_success,
    extract_failure,
    cast_success,
    cast_failure,
    from_iter,
    from_slice,
    from_object_slice,
    from_vec2,
    from_vec3
);
criterion_main!(benches);

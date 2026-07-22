use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numpy::{PyArrayLike1, PyArrayLike2, PyArrayLike3};
use pyo3::{
    ffi::c_str,
    types::{PyAnyMethods, PyDict},
    Python,
};

fn extract_array_like_1(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(5), 2_usize.pow(10), 2_usize.pow(15)];

    let mut group = c.benchmark_group("extract_array_like_1");
    for &size in SIZES {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("size", size).unwrap();

            let list = py
                .eval(
                    c_str!("[float(i) for i in range(size)]"),
                    Some(&locals),
                    None,
                )
                .unwrap();

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let list = black_box(&list);

                    let _array: PyArrayLike1<'_, f64> = black_box(list.extract().unwrap());
                });
            });
        });
    }
}

fn extract_array_like_2(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(3), 2_usize.pow(5), 2_usize.pow(8)];

    let mut group = c.benchmark_group("extract_array_like_2");
    for &size in SIZES {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("size", size).unwrap();

            let list = py
                .eval(
                    c_str!("[[float(i + j) for i in range(size)] for j in range(size)]"),
                    Some(&locals),
                    None,
                )
                .unwrap();

            group.throughput(Throughput::Elements(size.pow(2) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let list = black_box(&list);

                    let _array: PyArrayLike2<'_, f64> = black_box(list.extract().unwrap());
                });
            });
        });
    }
}

fn extract_array_like_3(c: &mut Criterion) {
    const SIZES: &[usize] = &[2_usize.pow(2), 2_usize.pow(4), 2_usize.pow(5)];

    let mut group = c.benchmark_group("extract_array_like_3");
    for &size in SIZES {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("size", size).unwrap();

            let list = py
                .eval(
                    c_str!("[[[float(i + j + k) for i in range(size)] for j in range(size)] for k in range(size)]"),
                    Some(&locals),
                    None,
                )
                .unwrap();

            group.throughput(Throughput::Elements(size.pow(3) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _size| {
                b.iter(|| {
                    let list = black_box(&list);

                    let _array: PyArrayLike3<'_, f64> = black_box(list.extract().unwrap());
                });
            });
        });
    }
}

criterion_group!(
    benches,
    extract_array_like_1,
    extract_array_like_2,
    extract_array_like_3
);
criterion_main!(benches);

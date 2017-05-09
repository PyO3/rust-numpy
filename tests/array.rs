
extern crate cpython;
extern crate numpy;

use numpy::*;

#[test]
fn new() {
    let gil = cpython::Python::acquire_gil();
    let n = 3;
    let m = 5;
    let arr = PyArray::new(gil.python(), &[n, m], NPY_TYPES::NPY_DOUBLE);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);
}

#[test]
fn zeros() {
    let gil = cpython::Python::acquire_gil();
    let n = 3;
    let m = 5;
    let arr = PyArray::zeros(gil.python(),
                             &[n, m],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::zeros(gil.python(),
                             &[n, m],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_FORTRANORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [8, n as isize * 8]);
}

#[test]
fn arange() {
    let gil = cpython::Python::acquire_gil();
    let arr = PyArray::arange(gil.python(), 0.0, 1.0, 0.1, NPY_TYPES::NPY_DOUBLE);
    println!("ndim = {:?}", arr.ndim());
    println!("dims = {:?}", arr.dims());
    println!("array = {:?}", arr.as_slice::<f64>());
}

#[test]
fn nonzero() {
    let gil = cpython::Python::acquire_gil();
    let z = PyArray::zeros(gil.python(),
                           &[5],
                           NPY_TYPES::NPY_DOUBLE,
                           NPY_ORDER::NPY_CORDER);
    let nz = PyArray::arange(gil.python(), 1.0, 2.0, 0.1, NPY_TYPES::NPY_DOUBLE);
    assert!(!z.nonzero()); // must be zero
    assert!(nz.nonzero()); // must be non-zero
}

#[test]
fn getitem() {
    let gil = cpython::Python::acquire_gil();
    let z = PyArray::arange(gil.python(), 1.0, 2.0, 0.1, NPY_TYPES::NPY_DOUBLE);
    let i = z.getitem(4);
    println!("{:?}", i);
}

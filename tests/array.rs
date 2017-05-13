
extern crate cpython;
extern crate numpy;

use numpy::*;

#[test]
fn new() {
    let gil = cpython::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let n = 3;
    let m = 5;
    let arr = PyArray::new::<f64>(gil.python(), &np, &[n, m]);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);
}

#[test]
fn zeros() {
    let gil = cpython::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let n = 3;
    let m = 5;
    let arr = PyArray::zeros(gil.python(),
                             &np,
                             &[n, m],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::zeros(gil.python(),
                             &np,
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
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::arange(gil.python(), &np, 0.0, 1.0, 0.1, NPY_TYPES::NPY_DOUBLE);
    println!("ndim = {:?}", arr.ndim());
    println!("dims = {:?}", arr.dims());
    println!("array = {:?}", arr.as_slice::<f64>().unwrap());
}

#[test]
fn as_array() {
    let gil = cpython::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::zeros(gil.python(),
                             &np,
                             &[3, 2, 4],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    let a = arr.as_array::<f64>().unwrap();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
               a.strides());
}

#[test]
#[should_panic]
fn as_array_panic() {
    let gil = cpython::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::zeros(gil.python(),
                             &np,
                             &[3, 2, 4],
                             NPY_TYPES::NPY_INT,
                             NPY_ORDER::NPY_CORDER);
    let _a = arr.as_array::<f32>().unwrap();
}

#[test]
fn into_pyarray() {
    let gil = cpython::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();

    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python(), &np);
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice::<i32>().unwrap());
}

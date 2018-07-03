extern crate pyo3;
extern crate ndarray;
extern crate numpy;

use ndarray::*;
use numpy::*;

#[test]
fn new() {
    let gil = pyo3::Python::acquire_gil();
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
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let n = 3;
    let m = 5;
    let arr = PyArray::zeros::<f64>(gil.python(), &np, &[n, m], NPY_CORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::zeros::<f64>(gil.python(), &np, &[n, m], NPY_FORTRANORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [8, n as isize * 8]);
}

#[test]
fn arange() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::arange::<f64>(gil.python(), &np, 0.0, 1.0, 0.1);
    println!("ndim = {:?}", arr.ndim());
    println!("dims = {:?}", arr.dims());
    println!("array = {:?}", arr.as_slice::<f64>().unwrap());
}

#[test]
fn as_array() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::zeros::<f64>(gil.python(), &np, &[3, 2, 4], NPY_CORDER);
    let a = arr.as_array::<f64>().unwrap();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(
        arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
        a.strides()
    );
}

#[test]
#[should_panic]
fn as_array_panic() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::zeros::<i32>(gil.python(), &np, &[3, 2, 4], NPY_CORDER);
    let _a = arr.as_array::<f32>().unwrap();
}

#[test]
fn into_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();

    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python(), &np);
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice::<i32>().unwrap());
    assert_eq!(arr.shape(), [3]);
}

#[test]
fn into_pyarray_array() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();

    let a = Array3::<f64>::zeros((3, 4, 2));
    let shape = a.shape().iter().cloned().collect::<Vec<_>>();
    let strides = a.strides().iter().map(|d| d * 8).collect::<Vec<_>>();
    println!("a.shape   = {:?}", a.shape());
    println!("a.strides = {:?}", a.strides());
    let pa = a.into_pyarray(gil.python(), &np);
    println!("pa.shape   = {:?}", pa.shape());
    println!("pa.strides = {:?}", pa.strides());
    assert_eq!(pa.shape(), shape);
    assert_eq!(pa.strides(), strides);
}

#[test]
fn iter_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = (0..10).map(|x| x * x).to_pyarray(gil.python(), &np);
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice::<i32>().unwrap());
    assert_eq!(arr.shape(), [10]);
}

#[test]
fn is_instance() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let np = PyArrayModule::import(py).unwrap();
    let arr = PyArray::new::<f64>(gil.python(), &np, &[3, 5]);
    assert!(py.is_instance::<PyArray, _>(&arr).unwrap());
    assert!(!py.is_instance::<pyo3::PyList, _>(&arr).unwrap());
}

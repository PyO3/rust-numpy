extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::*;
use numpy::*;

#[test]
fn new() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let n = 3;
    let m = 5;
    let arr = PyArray::<f64>::new(gil.python(), &np, &[n, m]);
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
    let arr = PyArray::<f64>::zeros(gil.python(), &np, &[n, m], NPY_CORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::<f64>::zeros(gil.python(), &np, &[n, m], NPY_FORTRANORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [8, n as isize * 8]);
}

#[test]
fn arange() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::<f64>::arange(gil.python(), &np, 0.0, 1.0, 0.1);
    println!("ndim = {:?}", arr.ndim());
    println!("dims = {:?}", arr.dims());
    println!("array = {:?}", arr.as_slice().unwrap());
}

#[test]
fn as_array() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = PyArray::<f64>::zeros(gil.python(), &np, &[3, 2, 4], NPY_CORDER);
    let a = arr.as_array().unwrap();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(
        arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
        a.strides()
    );
}

#[test]
fn into_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();

    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python(), &np);
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice().unwrap());
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
    assert_eq!(pa.shape(), shape.as_slice());
    assert_eq!(pa.strides(), strides.as_slice());
}

#[test]
fn iter_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let arr = (0..10).map(|x| x * x).to_pyarray(gil.python(), &np);
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice().unwrap());
    assert_eq!(arr.shape(), [10]);
}

#[test]
fn is_instance() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let np = PyArrayModule::import(py).unwrap();
    let arr = PyArray::<f64>::new(gil.python(), &np, &[3, 5]);
    assert!(py.is_instance::<PyArray<f64>, _>(&arr).unwrap());
    assert!(!py.is_instance::<pyo3::PyList, _>(&arr).unwrap());
}

#[test]
fn from_vec2() {
    let vec2 = vec![vec![1, 2, 3]; 2];
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let pyarray = PyArray::from_vec2(gil.python(), &np, &vec2).unwrap();
    assert_eq!(
        pyarray.as_array().unwrap(),
        array![[1, 2, 3], [1, 2, 3]].into_dyn()
    );
    assert!(PyArray::from_vec2(gil.python(), &np, &vec![vec![1], vec![2, 3]]).is_err());
}

#[test]
fn from_vec3() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let vec3 = vec![vec![vec![1, 2]; 2]; 2];
    let pyarray = PyArray::from_vec3(gil.python(), &np, &vec3).unwrap();
    assert_eq!(
        pyarray.as_array().unwrap(),
        array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]].into_dyn()
    );
}

#[test]
fn from_small_array() {
    let gil = pyo3::Python::acquire_gil();
    let np = PyArrayModule::import(gil.python()).unwrap();
    let array: [i32; 5] = [1, 2, 3, 4, 5];
    let pyarray = array.into_pyarray(gil.python(), &np);
    assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
}

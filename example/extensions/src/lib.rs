
#[macro_use]
extern crate cpython;
extern crate numpy;
extern crate ndarray;

use numpy::*;
use ndarray::*;
use cpython::{PyResult, Python};

// Pure Rust ndarray function
fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
    a * &x + &y
}

// Python-wrapper of `axpy`
fn axpy_py(py: Python, a: f64, x: PyArray, y: PyArray) -> PyResult<PyArray> {
    let np = PyArrayModule::import(py)?;
    let x = x.as_array().expect("x must be f64 array");
    let y = y.as_array().expect("y must be f64 array");
    Ok(axpy(a, x, y).into_pyarray(py, &np))
}

// Define module "_rust_ext"
py_module_initializer!(_rust_ext, init_rust_ext, PyInit__rust_ext, |py, m| {
    m.add(py, "__doc__", "Rust extension for NumPy")?;
    m.add(py,
             "axpy",
             py_fn!(py, axpy_py(a: f64, x: PyArray, y: PyArray)))?;
    Ok(())
});

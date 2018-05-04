#![feature(proc_macro, proc_macro_path_invoc, specialization)]

extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::*;
use numpy::*;
use pyo3::{py, PyModule, PyObject, PyResult, Python};

#[py::modinit(rust_ext)]
fn init_module(py: Python, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(py: Python, a: f64, x: PyArray, y: PyArray) -> PyResult<PyArray> {
        let np = PyArrayModule::import(py)?;
        let x = x.as_array().into_pyresult(py, "x must be f64 array")?;
        let y = y.as_array().into_pyresult(py, "y must be f64 array")?;
        Ok(axpy(a, x, y).into_pyarray(py, &np))
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(py: Python, a: f64, x: PyArray) -> PyResult<PyObject> {
        let x = x.as_array_mut().into_pyresult(py, "x must be f64 array")?;
        mult(a, x);
        Ok(py.None()) // Python function must returns
    }

    Ok(())
}

#![feature(use_extern_macros, specialization)]

extern crate ndarray;
extern crate numpy;

use ndarray::*;
use numpy::*;
use numpy::pyo3::prelude::*;

#[pymodinit]
fn rust_ext(py: Python, m: &PyModule) -> PyResult<()> {
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // You **must** write this sentence for PyArray type checker working correctly
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    let _np = PyArrayModule::import(py)?;

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
    fn axpy_py(py: Python, a: f64, x: &PyArray<f64>, y: &PyArray<f64>) -> PyResult<PyArray<f64>> {
        let np = PyArrayModule::import(py)?;
        let x = x.as_array().into_pyresult("x must be f64 array")?;
        let y = y.as_array().into_pyresult("y must be f64 array")?;
        Ok(axpy(a, x, y).into_pyarray(py, &np))
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArray<f64>) -> PyResult<()> {
        let x = x.as_array_mut().into_pyresult("x must be f64 array")?;
        mult(a, x);
        Ok(())
    }

    Ok(())
}

extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, IntoPyResult, PyArray, PyArrayModule};
use pyo3::prelude::{pymodinit, PyModule, PyResult, Python};

#[pymodinit]
fn rust_ext(py: Python, m: &PyModule) -> PyResult<()> {
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // You **must** write this statement for the PyArray type checker to work correctly
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
        Ok(axpy(a, x, y).into_pyarray(&np))
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

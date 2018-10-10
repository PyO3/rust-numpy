extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::prelude::{pymodinit, PyModule, PyResult, Python};

#[pymodinit]
fn rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
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
    fn axpy_py(
        py: Python,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> PyResult<PyArrayDyn<f64>> {
        let x = x.as_array()?;
        let y = y.as_array()?;
        Ok(axpy(a, x, y).to_pyarray(py).to_owned(py))
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut()?;
        mult(a, x);
        Ok(())
    }

    Ok(())
}

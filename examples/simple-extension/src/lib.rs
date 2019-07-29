extern crate ndarray;
extern crate ndarray_parallel;
extern crate numpy;
extern crate pyo3;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};
use ndarray::Zip;
use ndarray_parallel::prelude::*;

#[pymodule]
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
    ) -> Py<PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py).to_owned()
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut();
        mult(a, x);
        Ok(())
    }

    #[pyfn(m, "padd")]
    fn parallel_elementwise_add(py: Python, slf: &PyArray2<f64>, other: &PyArray2<f64>) {
        py.allow_threads(|| {
            Zip::from(slf.as_array_mut())
                .and(other.as_array())
                .par_apply(|s, &o| {
                    *s += o;
                });
        })
    }
    Ok(())
}

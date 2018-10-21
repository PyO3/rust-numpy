extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, IntoPyResult, PyArray1, PyArrayDyn, ToPyArray};
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
        // you can convert numpy error into PyErr via ?
        let x = x.as_array()?;
        // you can also specify your error context, via closure
        let y = y.as_array().into_pyresult_with(|| "y must be f64 array")?;
        Ok(axpy(a, x, y).to_pyarray(py).to_owned(py))
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut()?;
        mult(a, x);
        Ok(())
    }

    #[pyfn(m, "get_vec")]
    fn get_vec(py: Python, size: usize) -> PyResult<&PyArray1<f32>> {
        Ok(vec![0.0; size].into_pyarray(py))
    }
    // use numpy::slice_box::SliceBox;
    // #[pyfn(m, "get_slice")]
    // fn get_slice(py: Python, size: usize) -> PyResult<SliceBox<f32>> {
    //     let sbox = numpy::slice_box::SliceBox::new(vec![0.0; size].into_boxed_slice());
    //     Ok(sbox)
    // }
    Ok(())
}

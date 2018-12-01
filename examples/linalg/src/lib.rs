use ndarray_linalg::solve::Inverse;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::RuntimeError;
use pyo3::prelude::{pymodinit, Py, PyErr, PyModule, PyResult, Python};
use std::fmt::Display;

fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<RuntimeError, _>(format!("[rust_linalg] {}", e))
}

#[pymodinit]
fn rust_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "inv")]
    fn inv(py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array().inv().map_err(make_error)?;
        Ok(x.into_pyarray(py).to_owned())
    }
    Ok(())
}

use ndarray_linalg::solve::Inverse;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{pymodule, PyErr, PyModule, PyResult, Python};

#[pymodule]
fn rust_linalg(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn inv<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<&'py PyArray2<f64>> {
        let x = x
            .as_array()
            .inv()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("[rust_linalg] {}", e)))?;
        Ok(x.into_pyarray(py))
    }
    Ok(())
}

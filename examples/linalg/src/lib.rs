use ndarray_linalg::solve::Inverse;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn rust_linalg(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn inv<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
        let x = x.as_array();
        let y = x
            .inv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(y.into_pyarray(py))
    }
    Ok(())
}

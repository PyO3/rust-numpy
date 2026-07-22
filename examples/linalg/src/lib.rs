#[pyo3::pymodule]
mod rust_linalg {
    use ndarray_linalg::solve::Inverse;
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
    use pyo3::{exceptions::PyRuntimeError, pyfunction, Bound, PyResult, Python};

    #[pyfunction]
    fn inv<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = x.as_array();
        let y = x
            .inv()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(y.into_pyarray(py))
    }
}

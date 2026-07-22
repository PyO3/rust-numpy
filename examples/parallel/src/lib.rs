// We need to link `blas_src` directly, c.f. https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration
extern crate blas_src;

#[pyo3::pymodule]
mod rust_parallel {
    use numpy::ndarray::Zip;
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::{pyfunction, Bound, Python};

    #[pyfunction]
    fn rows_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = Zip::from(x.rows()).par_map_collect(|row| row.dot(&y));
        z.into_pyarray(py)
    }
}

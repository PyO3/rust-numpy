use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{
    pymodule,
    types::{PyDict, PyModule},
    PyResult, Python,
};

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // complex example
    fn conj(x: ArrayViewD<'_, Complex64>) -> ArrayD<Complex64> {
        x.map(|c| c.conj())
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'_, f64>,
        y: PyReadonlyArrayDyn<'_, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    // wrapper of `conj`
    #[pyfn(m)]
    #[pyo3(name = "conj")]
    fn conj_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'_, Complex64>,
    ) -> &'py PyArrayDyn<Complex64> {
        conj(x.as_array()).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "extract")]
    fn extract(d: &PyDict) -> f64 {
        let x = d
            .get_item("x")
            .unwrap()
            .downcast::<PyArray1<f64>>()
            .unwrap();

        x.readonly().as_array().sum()
    }

    Ok(())
}

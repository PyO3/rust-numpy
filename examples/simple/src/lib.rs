#[pyo3::pymodule]
mod rust_ext {
    use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
    use numpy::{
        datetime::{units, Timedelta},
        Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
        PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn,
    };
    use pyo3::{
        exceptions::PyIndexError,
        pyfunction,
        types::{PyDict, PyDictMethods},
        Bound, FromPyObject, Py, PyAny, PyResult, Python,
    };
    use std::ops::Add;

    // example using generic Py<PyAny>
    fn head(py: Python<'_>, x: ArrayViewD<'_, Py<PyAny>>) -> PyResult<Py<PyAny>> {
        x.get(0)
            .map(|obj| obj.clone_ref(py))
            .ok_or_else(|| PyIndexError::new_err("array index out of range"))
    }

    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // example using complex numbers
    fn conj(x: ArrayViewD<'_, Complex64>) -> ArrayD<Complex64> {
        x.map(|c| c.conj())
    }

    // example using generics
    fn generic_add<T: Copy + Add<Output = T>>(
        x: ArrayView1<'_, T>,
        y: ArrayView1<'_, T>,
    ) -> Array1<T> {
        &x + &y
    }

    // wrapper of `head`
    #[pyfunction(name = "head")]
    fn head_py<'py>(x: PyReadonlyArrayDyn<'py, Py<PyAny>>) -> PyResult<Py<PyAny>> {
        head(x.py(), x.as_array())
    }

    // wrapper of `axpy`
    #[pyfunction(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfunction(name = "mult")]
    fn mult_py<'py>(a: f64, mut x: PyReadwriteArrayDyn<'py, f64>) {
        let x = x.as_array_mut();
        mult(a, x);
    }

    // wrapper of `conj`
    #[pyfunction(name = "conj")]
    fn conj_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, Complex64>,
    ) -> Bound<'py, PyArrayDyn<Complex64>> {
        conj(x.as_array()).into_pyarray(py)
    }

    // example of how to extract an array from a dictionary
    #[pyfunction]
    fn extract(d: &Bound<'_, PyDict>) -> f64 {
        let x = d
            .get_item("x")
            .unwrap()
            .unwrap()
            .cast_into::<PyArray1<f64>>()
            .unwrap();

        x.readonly().as_array().sum()
    }

    // example using timedelta64 array
    #[pyfunction]
    fn add_minutes_to_seconds<'py>(
        mut x: PyReadwriteArray1<'py, Timedelta<units::Seconds>>,
        y: PyReadonlyArray1<'py, Timedelta<units::Minutes>>,
    ) {
        #[allow(deprecated)]
        Zip::from(x.as_array_mut())
            .and(y.as_array())
            .for_each(|x, y| *x = (i64::from(*x) + 60 * i64::from(*y)).into());
    }

    // This crate follows a strongly-typed approach to wrapping NumPy arrays
    // while Python API are often expected to work with multiple element types.
    //
    // That kind of limited polymorphis can be recovered by accepting an enumerated type
    // covering the supported element types and dispatching into a generic implementation.
    #[derive(FromPyObject)]
    enum SupportedArray<'py> {
        F64(Bound<'py, PyArray1<f64>>),
        I64(Bound<'py, PyArray1<i64>>),
    }

    #[pyfunction]
    fn polymorphic_add<'py>(
        x: SupportedArray<'py>,
        y: SupportedArray<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match (x, y) {
            (SupportedArray::F64(x), SupportedArray::F64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into_any()),
            (SupportedArray::I64(x), SupportedArray::I64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into_any()),
            (SupportedArray::F64(x), SupportedArray::I64(y))
            | (SupportedArray::I64(y), SupportedArray::F64(x)) => {
                let y = y.cast_array::<f64>(false)?;

                Ok(
                    generic_add(x.readonly().as_array(), y.readonly().as_array())
                        .into_pyarray(x.py())
                        .into_any(),
                )
            }
        }
    }
}

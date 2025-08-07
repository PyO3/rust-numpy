use ndarray::array;
use numpy::{get_array_module, AllowTypeChange, PyArrayLike1, PyArrayLike2, PyArrayLikeDyn};
use pyo3::{
    ffi::c_str,
    types::{IntoPyDict, PyAnyMethods, PyDict},
    Bound, Python,
};

fn get_np_locals(py: Python<'_>) -> Bound<'_, PyDict> {
    [("np", get_array_module(py).unwrap())]
        .into_py_dict(py)
        .unwrap()
}

#[test]
fn extract_reference() {
    Python::attach(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval(
                c_str!("np.array([[1,2],[3,4]], dtype='float64')"),
                Some(&locals),
                None,
            )
            .unwrap();
        let extracted_array = py_array.extract::<PyArrayLike2<'_, f64>>().unwrap();

        assert_eq!(
            array![[1_f64, 2_f64], [3_f64, 4_f64]],
            extracted_array.as_array()
        );
    });
}

#[test]
fn convert_array_on_extract() {
    Python::attach(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval(
                c_str!("np.array([[1,2],[3,4]], dtype='int32')"),
                Some(&locals),
                None,
            )
            .unwrap();
        let extracted_array = py_array
            .extract::<PyArrayLike2<'_, f64, AllowTypeChange>>()
            .unwrap();

        assert_eq!(
            array![[1_f64, 2_f64], [3_f64, 4_f64]],
            extracted_array.as_array()
        );
    });
}

#[test]
fn convert_list_on_extract() {
    Python::attach(|py| {
        let py_list = py
            .eval(c_str!("[[1.0,2.0],[3.0,4.0]]"), None, None)
            .unwrap();
        let extracted_array = py_list.extract::<PyArrayLike2<'_, f64>>().unwrap();

        assert_eq!(array![[1.0, 2.0], [3.0, 4.0]], extracted_array.as_array());
    });
}

#[test]
fn convert_array_in_list_on_extract() {
    Python::attach(|py| {
        let locals = get_np_locals(py);
        let py_array = py
            .eval(
                c_str!("[np.array([1.0, 2.0]), [3.0, 4.0]]"),
                Some(&locals),
                None,
            )
            .unwrap();
        let extracted_array = py_array.extract::<PyArrayLike2<'_, f64>>().unwrap();

        assert_eq!(array![[1.0, 2.0], [3.0, 4.0]], extracted_array.as_array());
    });
}

#[test]
fn convert_list_on_extract_dyn() {
    Python::attach(|py| {
        let py_list = py
            .eval(c_str!("[[[1,2],[3,4]],[[5,6],[7,8]]]"), None, None)
            .unwrap();
        let extracted_array = py_list
            .extract::<PyArrayLikeDyn<'_, i64, AllowTypeChange>>()
            .unwrap();

        assert_eq!(
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]].into_dyn(),
            extracted_array.as_array()
        );
    });
}

#[test]
fn convert_1d_list_on_extract() {
    Python::attach(|py| {
        let py_list = py.eval(c_str!("[1,2,3,4]"), None, None).unwrap();
        let extracted_array_1d = py_list.extract::<PyArrayLike1<'_, u32>>().unwrap();
        let extracted_array_dyn = py_list.extract::<PyArrayLikeDyn<'_, f64>>().unwrap();

        assert_eq!(array![1, 2, 3, 4], extracted_array_1d.as_array());
        assert_eq!(
            array![1_f64, 2_f64, 3_f64, 4_f64].into_dyn(),
            extracted_array_dyn.as_array()
        );
    });
}

#[test]
fn unsafe_cast_shall_fail() {
    Python::attach(|py| {
        let locals = get_np_locals(py);
        let py_list = py
            .eval(
                c_str!("np.array([1.1,2.2,3.3,4.4], dtype='float64')"),
                Some(&locals),
                None,
            )
            .unwrap();
        let extracted_array = py_list.extract::<PyArrayLike1<'_, i32>>();

        assert!(extracted_array.is_err());
    });
}

#[test]
fn unsafe_cast_with_coerce_works() {
    Python::attach(|py| {
        let locals = get_np_locals(py);
        let py_list = py
            .eval(
                c_str!("np.array([1.1,2.2,3.3,4.4], dtype='float64')"),
                Some(&locals),
                None,
            )
            .unwrap();
        let extracted_array = py_list
            .extract::<PyArrayLike1<'_, i32, AllowTypeChange>>()
            .unwrap();

        assert_eq!(array![1, 2, 3, 4], extracted_array.as_array());
    });
}

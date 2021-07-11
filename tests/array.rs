use ndarray::*;
use numpy::*;
use pyo3::{
    prelude::*,
    types::PyList,
    types::{IntoPyDict, PyDict},
};

fn get_np_locals(py: Python<'_>) -> &'_ PyDict {
    [("np", get_array_module(py).unwrap())].into_py_dict(py)
}

fn not_contiguous_array<'py>(py: Python<'py>) -> &'py PyArray1<i32> {
    py.eval(
        "np.array([1, 2, 3, 4], dtype='int32')[::2]",
        Some(get_np_locals(py)),
        None,
    )
    .unwrap()
    .downcast()
    .unwrap()
}

#[test]
fn new_c_order() {
    let dim = [3, 5];
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::new(py, dim, false);
        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dim);
        let size = std::mem::size_of::<f64>() as isize;
        assert!(arr.strides() == [dim[1] as isize * size, size]);
    })
}

#[test]
fn new_fortran_order() {
    let dim = [3, 5];
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::new(py, dim, true);
        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dim);
        let size = std::mem::size_of::<f64>() as isize;
        assert!(arr.strides() == [size, dim[0] as isize * size],);
    })
}

#[test]
fn tuple_as_dim() {
    let dim = (3, 5);
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::zeros(py, dim, false);
        assert!(arr.ndim() == 2);
        assert!(arr.dims() == [3, 5]);
    })
}

#[test]
fn zeros() {
    let shape = [3, 4];
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::zeros(py, shape, false);
        assert!(arr.ndim() == 2);
        assert!(arr.dims() == shape);
        assert!(arr.strides() == [shape[1] as isize * 8, 8]);

        let arr = PyArray::<f64, _>::zeros(py, shape, true);
        assert!(arr.ndim() == 2);
        assert!(arr.dims() == shape);
        assert!(arr.strides() == [8, shape[0] as isize * 8]);
    })
}

#[test]
fn arange() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::arange(py, 0.0, 1.0, 0.1);
        assert_eq!(arr.ndim(), 1);
        assert_eq!(arr.dims(), ndarray::Dim([10]));
    })
}

#[test]
fn as_array() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<f64, _>::zeros(py, [3, 2, 4], false);
        let arr = arr.readonly();
        let a = arr.as_array();
        assert_eq!(arr.shape(), a.shape());
        assert_eq!(
            arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
            a.strides()
        );
        let not_contiguous = not_contiguous_array(py).readonly();
        assert_eq!(not_contiguous.as_array(), array![1, 3]);
    })
}

#[test]
fn as_slice() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::<i32, _>::zeros(py, [3, 2, 4], false).readonly();
        assert_eq!(arr.as_slice().unwrap().len(), 3 * 2 * 4);
        let not_contiguous = not_contiguous_array(py).readonly();
        assert!(not_contiguous.as_slice().is_err());
    })
}

#[test]
fn is_instance() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray2::<f64>::new(py, [3, 5], false);
        assert!(arr.is_instance::<PyArray2<f64>>().unwrap());
        assert!(!arr.is_instance::<PyList>().unwrap());
    })
}

#[test]
fn from_string_slice() {
    let strs = vec!["a".to_string(), "😜".to_string()];
    pyo3::Python::with_gil(|py| {
        let pyarray = PyArray::from_slice(py, &strs);
        assert_eq!(
            pyarray.readonly().as_array(),
            array!["a".to_string(), "😜".to_string()]
        );
    })
}

#[test]
fn from_str_slice() {
    let strs = vec!["a", "😜"];
    pyo3::Python::with_gil(|py| {
        let pyarray = PyArray::from_slice(py, &strs);
        assert_eq!(pyarray.readonly().as_array(), array!["a", "😜"]);
    })
}

#[test]
fn from_vec2() {
    let vec2 = vec![vec![1, 2, 3]; 2];
    pyo3::Python::with_gil(|py| {
        let pyarray = PyArray::from_vec2(py, &vec2).unwrap();
        assert_eq!(pyarray.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
        assert!(PyArray::from_vec2(py, &[vec![1], vec![2, 3]]).is_err());
    })
}

#[test]
fn from_vec3() {
    let vec3 = vec![vec![vec![1, 2]; 2]; 2];
    pyo3::Python::with_gil(|py| {
        let pyarray = PyArray::from_vec3(py, &vec3).unwrap();
        assert_eq!(
            pyarray.readonly().as_array(),
            array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
        );
    })
}

#[test]
fn from_eval_to_fixed() {
    pyo3::Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let pyarray: &PyArray1<i32> = py
            .eval("np.array([1, 2, 3], dtype='int32')", Some(locals), None)
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(pyarray.readonly().as_array(), array![1, 2, 3]);
    })
}

#[test]
fn from_eval_to_dyn() {
    pyo3::Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let pyarray: &PyArrayDyn<i32> = py
            .eval(
                "np.array([[1, 2], [3, 4]], dtype='int32')",
                Some(locals),
                None,
            )
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(
            pyarray.readonly().as_array(),
            array![[1, 2], [3, 4]].into_dyn()
        );
    })
}

#[test]
fn from_eval_to_dyn_u64() {
    pyo3::Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let pyarray: &PyArrayDyn<u64> = py
            .eval(
                "np.array([[1, 2], [3, 4]], dtype='uint64')",
                Some(locals),
                None,
            )
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(
            pyarray.readonly().as_array(),
            array![[1, 2], [3, 4]].into_dyn()
        );
    })
}

#[test]
fn from_eval_fail_by_dtype() {
    pyo3::Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let converted: Result<&PyArray1<i32>, _> = py
            .eval("np.array([1, 2, 3], dtype='float64')", Some(locals), None)
            .unwrap()
            .extract();
        converted.unwrap_err().print_and_set_sys_last_vars(py);
    })
}

#[test]
fn from_eval_fail_by_dim() {
    pyo3::Python::with_gil(|py| {
        let locals = get_np_locals(py);
        let converted: Result<&PyArray2<i32>, _> = py
            .eval("np.array([1, 2, 3], dtype='int32')", Some(locals), None)
            .unwrap()
            .extract();
        converted.unwrap_err().print_and_set_sys_last_vars(py);
    })
}

#[test]
fn array_cast() {
    let vec2 = vec![vec![1.0, 2.0, 3.0]; 2];
    pyo3::Python::with_gil(|py| {
        let arr_f64 = PyArray::from_vec2(py, &vec2).unwrap();
        let arr_i32: &PyArray2<i32> = arr_f64.cast(false).unwrap();
        assert_eq!(arr_i32.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
    })
}

#[test]
fn handle_negative_strides() {
    let arr = array![[2, 3], [4, 5u32]];
    pyo3::Python::with_gil(|py| {
        let pyarr = arr.to_pyarray(py);
        let negstr_pyarr: &numpy::PyArray2<u32> = py
            .eval("a[::-1]", Some([("a", pyarr)].into_py_dict(py)), None)
            .unwrap()
            .downcast()
            .unwrap();
        assert_eq!(negstr_pyarr.to_owned_array(), arr.slice(s![..;-1, ..]));
    })
}

#[test]
fn dtype_from_py() {
    pyo3::Python::with_gil(|py| {
        let arr = array![[2, 3], [4, 5u32]];
        let pyarr = arr.to_pyarray(py);
        let dtype: &numpy::PyArrayDescr = py
            .eval("a.dtype", Some([("a", pyarr)].into_py_dict(py)), None)
            .unwrap()
            .downcast()
            .unwrap();
        assert_eq!(&format!("{:?}", dtype), "dtype('uint32')");
        assert_eq!(dtype.get_datatype().unwrap(), numpy::DataType::Uint32);
    })
}

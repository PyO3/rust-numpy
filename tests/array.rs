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
    let gil = pyo3::Python::acquire_gil();
    let dim = [3, 5];
    let arr = PyArray::<f64, _>::new(gil.python(), dim, false);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == dim);
    let size = std::mem::size_of::<f64>() as isize;
    assert!(arr.strides() == [dim[1] as isize * size, size]);
}

#[test]
fn new_fortran_order() {
    let gil = pyo3::Python::acquire_gil();
    let dim = [3, 5];
    let arr = PyArray::<f64, _>::new(gil.python(), dim, true);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == dim);
    let size = std::mem::size_of::<f64>() as isize;
    assert!(arr.strides() == [size, dim[0] as isize * size],);
}

#[test]
fn tuple_as_dim() {
    let gil = pyo3::Python::acquire_gil();
    let dim = (3, 5);
    let arr = PyArray::<f64, _>::zeros(gil.python(), dim, false);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [3, 5]);
}

#[test]
fn zeros() {
    let gil = pyo3::Python::acquire_gil();
    let n = 3;
    let m = 5;
    let arr = PyArray::<f64, _>::zeros(gil.python(), [n, m], false);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::<f64, _>::zeros(gil.python(), [n, m], true);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [8, n as isize * 8]);
}

#[test]
fn arange() {
    let gil = pyo3::Python::acquire_gil();
    let arr = PyArray::<f64, _>::arange(gil.python(), 0.0, 1.0, 0.1);
    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.dims(), ndarray::Dim([10]));
}

#[test]
fn as_array() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = PyArray::<f64, _>::zeros(py, [3, 2, 4], false);
    let arr = arr.readonly();
    let a = arr.as_array();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(
        arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
        a.strides()
    );
    let not_contiguous = not_contiguous_array(py).readonly();
    assert_eq!(not_contiguous.as_array(), array![1, 3])
}

#[test]
fn as_slice() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = PyArray::<i32, _>::zeros(py, [3, 2, 4], false).readonly();
    assert_eq!(arr.as_slice().unwrap().len(), 3 * 2 * 4);
    let not_contiguous = not_contiguous_array(py).readonly();
    assert!(not_contiguous.as_slice().is_err())
}

#[test]
fn is_instance() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = PyArray2::<f64>::new(gil.python(), [3, 5], false);
    assert!(py.is_instance::<PyArray2<f64>, _>(arr).unwrap());
    assert!(!py.is_instance::<PyList, _>(arr).unwrap());
}

#[test]
fn from_vec2() {
    let vec2 = vec![vec![1, 2, 3]; 2];
    let gil = pyo3::Python::acquire_gil();
    let pyarray = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    assert_eq!(pyarray.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
    assert!(PyArray::from_vec2(gil.python(), &[vec![1], vec![2, 3]]).is_err());
}

#[test]
fn from_vec3() {
    let gil = pyo3::Python::acquire_gil();
    let vec3 = vec![vec![vec![1, 2]; 2]; 2];
    let pyarray = PyArray::from_vec3(gil.python(), &vec3).unwrap();
    assert_eq!(
        pyarray.readonly().as_array(),
        array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
    );
}

#[test]
fn from_eval_to_fixed() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let locals = get_np_locals(py);
    let pyarray: &PyArray1<i32> = py
        .eval("np.array([1, 2, 3], dtype='int32')", Some(locals), None)
        .unwrap()
        .extract()
        .unwrap();
    assert_eq!(pyarray.readonly().as_array(), array![1, 2, 3]);
}

#[test]
fn from_eval_to_dyn() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
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
}

#[test]
fn from_eval_to_dyn_u64() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
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
}

#[test]
fn from_eval_fail_by_dtype() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let locals = get_np_locals(py);
    let converted: Result<&PyArray1<i32>, _> = py
        .eval("np.array([1, 2, 3], dtype='float64')", Some(locals), None)
        .unwrap()
        .extract();
    converted
        .unwrap_err()
        .print_and_set_sys_last_vars(gil.python());
}

#[test]
fn from_eval_fail_by_dim() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let locals = get_np_locals(py);
    let converted: Result<&PyArray2<i32>, _> = py
        .eval("np.array([1, 2, 3], dtype='int32')", Some(locals), None)
        .unwrap()
        .extract();
    converted
        .unwrap_err()
        .print_and_set_sys_last_vars(gil.python());
}

#[test]
fn array_cast() {
    let gil = pyo3::Python::acquire_gil();
    let vec2 = vec![vec![1.0, 2.0, 3.0]; 2];
    let arr_f64 = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    let arr_i32: &PyArray2<i32> = arr_f64.cast(false).unwrap();
    assert_eq!(arr_i32.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
}

#[test]
fn handle_negative_strides() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = array![[2, 3], [4, 5u32]];
    let pyarr = arr.to_pyarray(py);
    let negstr_pyarr: &numpy::PyArray2<u32> = py
        .eval("a[::-1]", Some([("a", pyarr)].into_py_dict(py)), None)
        .unwrap()
        .downcast()
        .unwrap();
    assert_eq!(negstr_pyarr.to_owned_array(), arr.slice(s![..;-1, ..]));
}

#[test]
fn dtype_from_py() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = array![[2, 3], [4, 5u32]];
    let pyarr = arr.to_pyarray(py);
    let dtype: &numpy::PyArrayDescr = py
        .eval("a.dtype", Some([("a", pyarr)].into_py_dict(py)), None)
        .unwrap()
        .downcast()
        .unwrap();
    assert_eq!(&format!("{:?}", dtype), "dtype('uint32')");
    assert_eq!(dtype.get_datatype().unwrap(), numpy::DataType::Uint32);
}

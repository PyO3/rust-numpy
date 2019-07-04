extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::*;
use numpy::*;
use pyo3::{
    prelude::*,
    types::PyList,
    types::{IntoPyDict, PyDict},
    AsPyPointer,
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
    .downcast_ref()
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
    let a = arr.as_array();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(
        arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
        a.strides()
    );
    let not_contiguous = not_contiguous_array(py);
    assert_eq!(not_contiguous.as_array(), array![1, 3])
}

#[test]
fn as_slice() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let arr = PyArray::<i32, _>::zeros(py, [3, 2, 4], false);
    assert_eq!(arr.as_slice().unwrap().len(), 3 * 2 * 4);
    let not_contiguous = not_contiguous_array(py);
    assert!(not_contiguous.as_slice().is_err())
}

#[test]
fn to_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();

    let a = vec![1, 2, 3];
    let arr = a.to_pyarray(gil.python());
    println!("arr.shape = {:?}", arr.shape());
    assert_eq!(arr.shape(), [3]);
    assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3])
}

#[test]
fn to_pyarray_array() {
    let gil = pyo3::Python::acquire_gil();

    let a = Array3::<f64>::zeros((3, 4, 2));
    let shape = a.shape().iter().cloned().collect::<Vec<_>>();
    let strides = a.strides().iter().map(|d| d * 8).collect::<Vec<_>>();
    println!("a.shape   = {:?}", a.shape());
    println!("a.strides = {:?}", a.strides());
    let pa = a.to_pyarray(gil.python());
    println!("pa.shape   = {:?}", pa.shape());
    println!("pa.strides = {:?}", pa.strides());
    assert_eq!(pa.shape(), shape.as_slice());
    assert_eq!(pa.strides(), strides.as_slice());
}

#[test]
fn iter_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let arr = PyArray::from_iter(gil.python(), (0..10).map(|x| x * x));
    assert_eq!(
        arr.as_slice().unwrap(),
        &[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    );
}

#[test]
fn long_iter_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let arr = PyArray::from_iter(gil.python(), (0u32..512).map(|x| x));
    let slice = arr.as_slice().unwrap();
    for (i, &elem) in slice.iter().enumerate() {
        assert_eq!(i as u32, elem);
    }
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
    assert_eq!(pyarray.as_array(), array![[1, 2, 3], [1, 2, 3]]);
    assert!(PyArray::from_vec2(gil.python(), &vec![vec![1], vec![2, 3]]).is_err());
}

#[test]
fn from_vec3() {
    let gil = pyo3::Python::acquire_gil();
    let vec3 = vec![vec![vec![1, 2]; 2]; 2];
    let pyarray = PyArray::from_vec3(gil.python(), &vec3).unwrap();
    assert_eq!(
        pyarray.as_array(),
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
    assert_eq!(pyarray.as_array(), array![1, 2, 3]);
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
    assert_eq!(pyarray.as_array(), array![[1, 2], [3, 4]].into_dyn());
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
    assert_eq!(pyarray.as_array(), array![[1, 2], [3, 4]].into_dyn());
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

macro_rules! small_array_test {
    ($($t: ident)+) => {
        #[test]
        fn from_small_array() {
            let gil = pyo3::Python::acquire_gil();
            $({
                let array: [$t; 2] = [$t::min_value(), $t::max_value()];
                let pyarray = array.to_pyarray(gil.python());
                assert_eq!(
                    pyarray.as_slice().unwrap(),
                    &[$t::min_value(), $t::max_value()]
                );
            })+
        }
    };
}

small_array_test!(i8 u8 i16 u16 i32 u32 i64 u64 usize);

#[test]
fn array_usize_dtype() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();

    let a: Vec<usize> = vec![1, 2, 3];
    let x = a.into_pyarray(py);
    let x_repr = format!("{:?}", x);

    let x_repr_expected = if cfg!(target_pointer_width = "64") {
        "array([1, 2, 3], dtype=uint64)"
    } else {
        "array([1, 2, 3], dtype=uint32)"
    };
    assert_eq!(x_repr, x_repr_expected);
}

#[test]
fn array_cast() {
    let gil = pyo3::Python::acquire_gil();
    let vec2 = vec![vec![1.0, 2.0, 3.0]; 2];
    let arr_f64 = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    let arr_i32: &PyArray2<i32> = arr_f64.cast(false).unwrap();
    assert_eq!(arr_i32.as_array(), array![[1, 2, 3], [1, 2, 3]]);
}

#[test]
fn into_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();
    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python());
    assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3])
}

#[test]
fn into_pyarray_array() {
    let gil = pyo3::Python::acquire_gil();
    let arr = Array3::<f64>::zeros((3, 4, 2));
    let shape = arr.shape().iter().cloned().collect::<Vec<_>>();
    let strides = arr.strides().iter().map(|d| d * 8).collect::<Vec<_>>();
    let py_arr = arr.into_pyarray(gil.python());
    assert_eq!(py_arr.shape(), shape.as_slice());
    assert_eq!(py_arr.strides(), strides.as_slice());
}

#[test]
fn into_pyarray_cant_resize() {
    let gil = pyo3::Python::acquire_gil();
    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python());
    assert!(arr.resize(100).is_err())
}

// TODO: Replace it by pyo3::py_run when https://github.com/PyO3/pyo3/pull/512 is released.
macro_rules! py_run {
    ($py:expr, $val:expr, $code:expr) => {{
        let d = pyo3::types::PyDict::new($py);
        d.set_item(stringify!($val), &$val).unwrap();
        $py.run($code, None, Some(d))
            .map_err(|e| {
                e.print($py);
                $py.run("import sys; sys.stderr.flush()", None, None)
                    .unwrap();
            })
            .expect($code)
    }};
}

#[test]
fn into_obj_vec_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let dict = PyDict::new(py);
    let string = pyo3::types::PyString::new(py, "Hello python :)");
    let a = vec![dict.as_ptr(), string.as_ptr()];
    let arr = a.into_pyarray(py);
    py_run!(py, arr, "assert arr[0] == {}");
    py_run!(py, arr, "assert arr[1] == 'Hello python :)'");
}

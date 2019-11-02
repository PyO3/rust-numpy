use ndarray::*;
use numpy::*;
use pyo3::{py_run, types::PyDict, ToPyObject};

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
fn usize_dtype() {
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

#[test]
fn into_obj_vec_to_pyarray() {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let dict = PyDict::new(py);
    let string = pyo3::types::PyString::new(py, "Hello python :)");
    let a = vec![dict.to_object(py), string.to_object(py)];
    let arr = a.into_pyarray(py);
    py_run!(py, arr, "print(arr)");
    py_run!(py, arr, "assert arr[1] == 'Hello python :)'");
}

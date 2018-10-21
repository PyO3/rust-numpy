extern crate ndarray;
extern crate numpy;
extern crate pyo3;

use ndarray::*;
use numpy::*;
use pyo3::prelude::*;

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
    println!("ndim = {:?}", arr.ndim());
    println!("dims = {:?}", arr.dims());
    println!("array = {:?}", arr.as_slice().unwrap());
}

#[test]
fn as_array() {
    let gil = pyo3::Python::acquire_gil();
    let arr = PyArray::<f64, _>::zeros(gil.python(), [3, 2, 4], false);
    let a = arr.as_array().unwrap();
    assert_eq!(arr.shape(), a.shape());
    assert_eq!(
        arr.strides().iter().map(|x| x / 8).collect::<Vec<_>>(),
        a.strides()
    );
}

#[test]
fn to_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();

    let a = vec![1, 2, 3];
    let arr = a.to_pyarray(gil.python());
    println!("arr.shape = {:?}", arr.shape());
    println!("arr = {:?}", arr.as_slice().unwrap());
    assert_eq!(arr.shape(), [3]);
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
    assert!(!py.is_instance::<pyo3::PyList, _>(arr).unwrap());
}

#[test]
fn from_vec2() {
    let vec2 = vec![vec![1, 2, 3]; 2];
    let gil = pyo3::Python::acquire_gil();
    let pyarray = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    assert_eq!(pyarray.as_array().unwrap(), array![[1, 2, 3], [1, 2, 3]]);
    assert!(PyArray::from_vec2(gil.python(), &vec![vec![1], vec![2, 3]]).is_err());
}

#[test]
fn from_vec3() {
    let gil = pyo3::Python::acquire_gil();
    let vec3 = vec![vec![vec![1, 2]; 2]; 2];
    let pyarray = PyArray::from_vec3(gil.python(), &vec3).unwrap();
    assert_eq!(
        pyarray.as_array().unwrap(),
        array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
    );
}

#[test]
fn from_eval() {
    let gil = pyo3::Python::acquire_gil();
    let np = get_array_module(gil.python()).unwrap();
    let dict = PyDict::new(gil.python());
    dict.set_item("np", np).unwrap();
    let pyarray: &PyArray1<i32> = gil
        .python()
        .eval("np.array([1, 2, 3], dtype='int32')", Some(&dict), None)
        .unwrap()
        .extract()
        .unwrap();
    assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3]);
}

#[test]
fn from_eval_fail() {
    let gil = pyo3::Python::acquire_gil();
    let np = get_array_module(gil.python()).unwrap();
    let dict = PyDict::new(gil.python());
    dict.set_item("np", np).unwrap();
    let converted: Result<&PyArray1<i32>, _> = gil
        .python()
        .eval("np.array([1, 2, 3], dtype='float64')", Some(&dict), None)
        .unwrap()
        .extract();
    assert!(converted.is_err());
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

small_array_test!(i8 u8 i16 u16 i32 u32 i64 u64);

#[test]
fn array_cast() {
    let gil = pyo3::Python::acquire_gil();
    let vec2 = vec![vec![1.0, 2.0, 3.0]; 2];
    let arr_f64 = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    let arr_i32: &PyArray2<i32> = arr_f64.cast(false).unwrap();
    assert_eq!(arr_i32.as_array().unwrap(), array![[1, 2, 3], [1, 2, 3]]);
}

#[test]
fn into_pyarray_vec() {
    let gil = pyo3::Python::acquire_gil();
    let a = vec![1, 2, 3];
    let arr = a.into_pyarray(gil.python());
    assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3])
}

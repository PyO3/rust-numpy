use ndarray::*;
use numpy::*;

#[test]
fn to_pyarray_vec() {
    pyo3::Python::with_gil(|py| {
        let a = vec![1, 2, 3];
        let arr = a.to_pyarray(py).readonly();
        println!("arr.shape = {:?}", arr.shape());
        assert_eq!(arr.shape(), [3]);
        assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3])
    })
}

#[test]
fn to_pyarray_array() {
    pyo3::Python::with_gil(|py| {
        let a = Array3::<f64>::zeros((3, 4, 2));
        let shape = a.shape().to_vec();
        let strides = a.strides().iter().map(|d| d * 8).collect::<Vec<_>>();
        println!("a.shape   = {:?}", a.shape());
        println!("a.strides = {:?}", a.strides());
        let pa = a.to_pyarray(py);
        println!("pa.shape   = {:?}", pa.shape());
        println!("pa.strides = {:?}", pa.strides());
        assert_eq!(pa.shape(), shape.as_slice());
        assert_eq!(pa.strides(), strides.as_slice());
    })
}

#[test]
fn iter_to_pyarray() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::from_iter(py, (0..10).map(|x| x * x)).readonly();
        assert_eq!(
            arr.as_slice().unwrap(),
            &[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        );
    })
}

#[test]
fn long_iter_to_pyarray() {
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::from_iter(py, 0u32..512).readonly();
        let slice = arr.as_slice().unwrap();
        for (i, &elem) in slice.iter().enumerate() {
            assert_eq!(i as u32, elem);
        }
    })
}

macro_rules! small_array_test {
    ($($t: ident)+) => {
        #[test]
        fn from_small_array() {
            $({
                pyo3::Python::with_gil(|py| {
                    let array: [$t; 2] = [$t::min_value(), $t::max_value()];
                    let pyarray = array.to_pyarray(py).readonly();
                    assert_eq!(
                        pyarray.as_slice().unwrap(),
                        &[$t::min_value(), $t::max_value()]
                    );
                })
            })+
        }
    };
}

small_array_test!(i8 u8 i16 u16 i32 u32 i64 u64 usize);

#[test]
fn usize_dtype() {
    let a: Vec<usize> = vec![1, 2, 3];

    pyo3::Python::with_gil(|py| {
        let x = a.into_pyarray(py);

        if cfg!(target_pointer_width = "64") {
            pyo3::py_run!(py, x, "assert str(x.dtype) == 'uint64'")
        } else {
            pyo3::py_run!(py, x, "assert str(x.dtype) == 'uint32'")
        };
    })
}

#[test]
fn into_pyarray_vec() {
    let a = vec![1, 2, 3];
    pyo3::Python::with_gil(|py| {
        let arr = a.into_pyarray(py).readonly();
        assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3])
    })
}

#[test]
fn into_pyarray_array() {
    let arr = Array3::<f64>::zeros((3, 4, 2));
    let shape = arr.shape().to_vec();
    let strides = arr.strides().iter().map(|d| d * 8).collect::<Vec<_>>();
    pyo3::Python::with_gil(|py| {
        let py_arr = arr.into_pyarray(py);
        assert_eq!(py_arr.shape(), shape.as_slice());
        assert_eq!(py_arr.strides(), strides.as_slice());
    })
}

#[test]
fn into_pyarray_cant_resize() {
    let a = vec![1, 2, 3];
    pyo3::Python::with_gil(|py| {
        let arr = a.into_pyarray(py);
        assert!(arr.resize(100).is_err())
    })
}

#[test]
fn forder_to_pyarray() {
    pyo3::Python::with_gil(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let fortran_matrix = matrix.reversed_axes();
        let fmat_py = fortran_matrix.to_pyarray(py);
        assert_eq!(
            fmat_py.readonly().as_array(),
            array![[0, 2, 4, 6], [1, 3, 5, 7]],
        );
        pyo3::py_run!(py, fmat_py, "assert fmat_py.flags['F_CONTIGUOUS']")
    })
}

#[test]
fn slice_to_pyarray() {
    pyo3::Python::with_gil(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let slice = matrix.slice(s![1..4; -1, ..]);
        let slice_py = slice.to_pyarray(py);
        assert_eq!(
            slice_py.readonly().as_array(),
            array![[6, 7], [4, 5], [2, 3]],
        );
        pyo3::py_run!(py, slice_py, "assert slice_py.flags['C_CONTIGUOUS']")
    })
}

#[test]
fn forder_into_pyarray() {
    pyo3::Python::with_gil(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let fortran_matrix = matrix.reversed_axes();
        let fmat_py = fortran_matrix.into_pyarray(py);
        assert_eq!(
            fmat_py.readonly().as_array(),
            array![[0, 2, 4, 6], [1, 3, 5, 7]],
        );
        pyo3::py_run!(py, fmat_py, "assert fmat_py.flags['F_CONTIGUOUS']")
    })
}

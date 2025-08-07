use std::cmp::Ordering;
use std::mem::size_of;

use ndarray::{array, s, Array2, Array3};
use numpy::{prelude::*, PyArray};
use pyo3::{
    py_run,
    types::{PyAnyMethods, PyDict, PyString},
    Python,
};

#[test]
fn to_pyarray_vec() {
    Python::attach(|py| {
        #[allow(clippy::useless_vec)]
        let arr = vec![1, 2, 3].to_pyarray(py);

        assert_eq!(arr.shape(), [3]);
        assert_eq!(arr.readonly().as_slice().unwrap(), &[1, 2, 3])
    });
}

#[test]
fn to_pyarray_boxed_slice() {
    Python::attach(|py| {
        let arr = vec![1, 2, 3].into_boxed_slice().to_pyarray(py);

        assert_eq!(arr.shape(), [3]);
        assert_eq!(arr.readonly().as_slice().unwrap(), &[1, 2, 3])
    });
}

#[test]
fn to_pyarray_array() {
    Python::attach(|py| {
        let arr = Array3::<f64>::zeros((3, 4, 2));

        let shape = arr.shape().to_vec();
        let strides = arr
            .strides()
            .iter()
            .map(|dim| dim * size_of::<f64>() as isize)
            .collect::<Vec<_>>();

        let py_arr = PyArray::from_array(py, &arr);

        assert_eq!(py_arr.shape(), shape.as_slice());
        assert_eq!(py_arr.strides(), strides.as_slice());
    });
}

#[test]
fn iter_to_pyarray() {
    Python::attach(|py| {
        let arr = PyArray::from_iter(py, (0..10).map(|x| x * x));

        assert_eq!(
            arr.readonly().as_slice().unwrap(),
            &[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        );
    });
}

#[test]
fn long_iter_to_pyarray() {
    Python::attach(|py| {
        let arr = PyArray::from_iter(py, 0_u32..512);

        assert_eq!(
            arr.readonly().as_slice().unwrap(),
            (0_u32..512).collect::<Vec<_>>(),
        );
    });
}

#[test]
fn from_small_array() {
    macro_rules! small_array_test {
        ($($t:ty)+) => {
            $({
                Python::attach(|py| {
                    let array: [$t; 2] = [<$t>::MIN, <$t>::MAX];
                    let pyarray = array.to_pyarray(py);

                    assert_eq!(
                        pyarray.readonly().as_slice().unwrap(),
                        &[<$t>::MIN, <$t>::MAX]
                    );
                });
            })+
        };
    }

    small_array_test!(i8 u8 i16 u16 i32 u32 i64 u64 isize usize);
}

#[test]
fn usize_dtype() {
    Python::attach(|py| {
        let x = vec![1_usize, 2, 3].into_pyarray(py);

        if cfg!(target_pointer_width = "64") {
            py_run!(py, x, "assert str(x.dtype) == 'uint64'")
        } else {
            py_run!(py, x, "assert str(x.dtype) == 'uint32'")
        };
    })
}

#[test]
fn into_pyarray_vec() {
    Python::attach(|py| {
        let arr = vec![1, 2, 3].into_pyarray(py);

        assert_eq!(arr.readonly().as_slice().unwrap(), &[1, 2, 3])
    });
}

#[test]
fn into_pyarray_boxed_slice() {
    Python::attach(|py| {
        let arr = vec![1, 2, 3].into_boxed_slice().into_pyarray(py);

        assert_eq!(arr.readonly().as_slice().unwrap(), &[1, 2, 3])
    });
}

#[test]
fn into_pyarray_array() {
    Python::attach(|py| {
        let arr = Array3::<f64>::zeros((3, 4, 2));

        let shape = arr.shape().to_vec();
        let strides = arr
            .strides()
            .iter()
            .map(|dim| dim * size_of::<f64>() as isize)
            .collect::<Vec<_>>();

        let py_arr = arr.into_pyarray(py);

        assert_eq!(py_arr.shape(), shape.as_slice());
        assert_eq!(py_arr.strides(), strides.as_slice());
    });
}

#[test]
fn into_pyarray_cannot_resize() {
    Python::attach(|py| {
        let arr = vec![1, 2, 3].into_pyarray(py);

        unsafe {
            assert!(arr.resize(100).is_err());
        }
    });
}

#[test]
fn into_pyarray_can_write() {
    Python::attach(|py| {
        let arr = vec![1, 2, 3].into_pyarray(py);

        py_run!(py, arr, "assert arr.flags['WRITEABLE']");
        py_run!(py, arr, "arr[1] = 4");
    });
}

#[test]
fn collapsed_into_pyarray() {
    // Check that `into_pyarray` works for array with the pointer of the first element is
    // not at the start of the allocation.
    // See https://github.com/PyO3/rust-numpy/issues/182 for more.
    Python::attach(|py| {
        let mut arr = Array2::<f64>::from_shape_fn([3, 4], |(i, j)| (i * 10 + j) as f64);
        arr.slice_collapse(s![1.., ..]);
        let cloned_arr = arr.clone();

        let py_arr = arr.into_pyarray(py);

        assert_eq!(py_arr.readonly().as_array(), cloned_arr);
    });
}

#[test]
fn sliced_to_pyarray() {
    Python::attach(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let sliced_matrix = matrix.slice(s![1..4; -1, ..]);

        let py_arr = sliced_matrix.to_pyarray(py);

        assert_eq!(py_arr.readonly().as_array(), array![[6, 7], [4, 5], [2, 3]],);

        py_run!(py, py_arr, "assert py_arr.flags['C_CONTIGUOUS']")
    });
}

#[test]
fn forder_to_pyarray() {
    Python::attach(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let forder_matrix = matrix.reversed_axes();

        let py_arr = forder_matrix.to_pyarray(py);

        assert_eq!(
            py_arr.readonly().as_array(),
            array![[0, 2, 4, 6], [1, 3, 5, 7]],
        );

        py_run!(py, py_arr, "assert py_arr.flags['F_CONTIGUOUS']")
    });
}

#[test]
fn forder_into_pyarray() {
    Python::attach(|py| {
        let matrix = Array2::from_shape_vec([4, 2], vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let forder_matrix = matrix.reversed_axes();

        let py_arr = forder_matrix.into_pyarray(py);

        assert_eq!(
            py_arr.readonly().as_array(),
            array![[0, 2, 4, 6], [1, 3, 5, 7]],
        );

        py_run!(py, py_arr, "assert py_arr.flags['F_CONTIGUOUS']")
    });
}

#[test]
fn to_pyarray_object_vec() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        let string = PyString::new(py, "Hello:)");
        #[allow(clippy::useless_vec)] // otherwise we do not test the right trait impl
        let vec = vec![dict.into_any().unbind(), string.into_any().unbind()];

        let arr = vec.to_pyarray(py);

        for (a, b) in vec.iter().zip(arr.readonly().as_slice().unwrap().iter()) {
            assert_eq!(
                a.bind(py).compare(b).map_err(|e| e.print(py)).unwrap(),
                Ordering::Equal
            );
        }
    });
}

#[test]
fn to_pyarray_object_array() {
    Python::attach(|py| {
        let mut nd_arr = Array2::from_shape_fn((2, 3), |(_, _)| py.None());
        nd_arr[(0, 2)] = PyDict::new(py).into_any().unbind();
        nd_arr[(1, 0)] = PyString::new(py, "Hello:)").into_any().unbind();

        let py_arr = nd_arr.to_pyarray(py);

        for (a, b) in nd_arr
            .as_slice()
            .unwrap()
            .iter()
            .zip(py_arr.readonly().as_slice().unwrap().iter())
        {
            assert_eq!(
                a.bind(py).compare(b).map_err(|e| e.print(py)).unwrap(),
                Ordering::Equal
            );
        }
    });
}

#[test]
fn slice_container_type_confusion() {
    Python::attach(|py| {
        let mut nd_arr = Array2::from_shape_fn((2, 3), |(_, _)| py.None());
        nd_arr[(0, 2)] = PyDict::new(py).into_any().unbind();
        nd_arr[(1, 0)] = PyString::new(py, "Hello:)").into_any().unbind();

        let _py_arr = nd_arr.into_pyarray(py);

        // Dropping `_py_arr` used to trigger a segmentation fault due to calling `Py_DECREF`
        // on 1, 2 and 3 interpreted as pointers into the Python heap
        // after having created a `SliceBox<PyObject>` backing `_py_arr`,
        // c.f. https://github.com/PyO3/rust-numpy/issues/232.
        let _py_arr = vec![1, 2, 3].into_pyarray(py);
    });
}

#[cfg(feature = "nalgebra")]
#[test]
fn matrix_to_numpy() {
    let matrix = nalgebra::Matrix3::<i32>::new(0, 1, 2, 3, 4, 5, 6, 7, 8);
    assert!(nalgebra::RawStorage::is_contiguous(&matrix.data));

    Python::attach(|py| {
        let array = matrix.to_pyarray(py);

        assert_eq!(
            array.readonly().as_array(),
            array![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        );
    });

    let matrix = matrix.row(0);
    assert!(!nalgebra::RawStorage::is_contiguous(&matrix.data));

    Python::attach(|py| {
        let array = matrix.to_pyarray(py);

        assert_eq!(array.readonly().as_array(), array![[0, 1, 2]]);
    });

    let vector = nalgebra::Vector4::<i32>::new(-4, 1, 2, 3);

    Python::attach(|py| {
        let array = vector.to_pyarray(py);

        assert_eq!(array.readonly().as_array(), array![[-4], [1], [2], [3]]);
    });

    let vector = nalgebra::RowVector2::<i32>::new(23, 42);

    Python::attach(|py| {
        let array = vector.to_pyarray(py);

        assert_eq!(array.readonly().as_array(), array![[23, 42]]);
    });
}

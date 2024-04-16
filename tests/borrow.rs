use std::thread::spawn;

use numpy::{
    array::PyArrayMethods, npyffi::NPY_ARRAY_WRITEABLE, PyArray, PyArray1, PyArray2,
    PyReadonlyArray3, PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::{
    py_run, pyclass, pymethods,
    types::{IntoPyDict, PyAnyMethods},
    Py, Python,
};

#[test]
fn distinct_borrows() {
    Python::with_gil(|py| {
        let array1 = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);
        let array2 = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let exclusive1 = array1.readwrite();
        let exclusive2 = array2.readwrite();

        assert_eq!(exclusive2.shape(), [1, 2, 3]);
        assert_eq!(exclusive1.shape(), [1, 2, 3]);
    });
}

#[test]
fn multiple_shared_borrows() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let shared1 = array.readonly();
        let shared2 = array.readonly();

        assert_eq!(shared2.shape(), [1, 2, 3]);
        assert_eq!(shared1.shape(), [1, 2, 3]);
    });
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn exclusive_and_shared_borrows() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let _exclusive = array.readwrite();
        let _shared = array.readonly();
    });
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn shared_and_exclusive_borrows() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let _shared = array.readonly();
        let _exclusive = array.readwrite();
    });
}

#[test]
fn multiple_exclusive_borrows() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let _exclusive = array.try_readwrite().unwrap();

        let err = array.try_readwrite().unwrap_err();
        assert_eq!(err.to_string(), "The given array is already borrowed");
    });
}

#[test]
fn exclusive_borrow_requires_writeable() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        unsafe {
            (*array.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE;
        }

        let err = array.try_readwrite().unwrap_err();
        assert_eq!(err.to_string(), "The given array is not writeable");
    });
}

#[pyclass]
struct Borrower;

#[pymethods]
impl Borrower {
    fn shared(&self, _array: PyReadonlyArray3<'_, f64>) {}

    fn exclusive(&self, _array: PyReadwriteArray3<'_, f64>) {}
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn borrows_span_frames() {
    Python::with_gil(|py| {
        let borrower = Py::new(py, Borrower).unwrap();

        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let _exclusive = array.readwrite();

        py_run!(py, borrower array, "borrower.exclusive(array)");
    });
}

#[test]
fn borrows_span_threads() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let _exclusive = array.readwrite();

        let array = array.unbind();

        py.allow_threads(move || {
            let thread = spawn(move || {
                Python::with_gil(|py| {
                    let array = array.bind(py);

                    let _exclusive = array.readwrite();
                });
            });

            assert!(thread.join().is_err());
        });
    });
}

#[test]
fn shared_borrows_can_be_cloned() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);

        let shared1 = array.readonly();
        let shared2 = shared1.clone();

        assert_eq!(shared2.shape(), [1, 2, 3]);
        assert_eq!(shared1.shape(), [1, 2, 3]);
    });
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn overlapping_views_conflict() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);
        let locals = [("array", array)].into_py_dict_bound(py);

        let view1 = py
            .eval_bound("array[0,0,0:2]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view1.shape(), [2]);

        let view2 = py
            .eval_bound("array[0,0,1:3]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [2]);

        let _exclusive1 = view1.readwrite();
        let _exclusive2 = view2.readwrite();
    });
}

#[test]
fn non_overlapping_views_do_not_conflict() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);
        let locals = [("array", array)].into_py_dict_bound(py);

        let view1 = py
            .eval_bound("array[0,0,0:1]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view1.shape(), [1]);

        let view2 = py
            .eval_bound("array[0,0,2:3]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [1]);

        let exclusive1 = view1.readwrite();
        let exclusive2 = view2.readwrite();

        assert_eq!(exclusive2.len(), 1);
        assert_eq!(exclusive1.len(), 1);
    });
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn conflict_due_to_overlapping_views() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, 3, false);
        let locals = [("array", array)].into_py_dict_bound(py);

        let view1 = py
            .eval_bound("array[0:2]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view1.shape(), [2]);

        let view2 = py
            .eval_bound("array[1:3]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [2]);

        let _exclusive1 = view1.readwrite();
        let _shared2 = view2.readonly();
    });
}

#[test]
#[should_panic(expected = "AlreadyBorrowed")]
fn conflict_due_to_reborrow_of_overlapping_views() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, 3, false);
        let locals = [("array", array)].into_py_dict_bound(py);

        let view1 = py
            .eval_bound("array[0:2]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view1.shape(), [2]);

        let view2 = py
            .eval_bound("array[1:3]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray1<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [2]);

        let shared1 = view1.readonly();
        let _shared2 = view2.readonly();

        drop(shared1);
        let _exclusive1 = view1.readwrite();
    });
}

#[test]
fn interleaved_views_do_not_conflict() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (23, 42, 3), false);
        let locals = [("array", array)].into_py_dict_bound(py);

        let view1 = py
            .eval_bound("array[:,:,0]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray2<f64>>()
            .unwrap();
        assert_eq!(view1.shape(), [23, 42]);

        let view2 = py
            .eval_bound("array[:,:,1]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray2<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [23, 42]);

        let view3 = py
            .eval_bound("array[:,:,2]", None, Some(&locals))
            .unwrap()
            .downcast_into::<PyArray2<f64>>()
            .unwrap();
        assert_eq!(view2.shape(), [23, 42]);

        let exclusive1 = view1.readwrite();
        let exclusive2 = view2.readwrite();
        let exclusive3 = view3.readwrite();

        assert_eq!(exclusive3.len(), 23 * 42);
        assert_eq!(exclusive2.len(), 23 * 42);
        assert_eq!(exclusive1.len(), 23 * 42);
    });
}

#[test]
fn extract_readonly() {
    Python::with_gil(|py| {
        let ob = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false).into_any();
        ob.extract::<PyReadonlyArray3<'_, f64>>().unwrap();
    });
}

#[test]
fn extract_readwrite() {
    Python::with_gil(|py| {
        let ob = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false).into_any();
        ob.extract::<PyReadwriteArray3<'_, f64>>().unwrap();
    });
}

#[test]
fn readonly_as_array_slice_get() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);
        let array = array.readonly();

        assert_eq!(array.as_array().shape(), [1, 2, 3]);
        assert_eq!(array.as_slice().unwrap().len(), 2 * 3);
        assert_eq!(*array.get([0, 1, 2]).unwrap(), 0.0);
    });
}

#[test]
fn readwrite_as_array_slice() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, (1, 2, 3), false);
        let mut array = array.readwrite();

        assert_eq!(array.as_array().shape(), [1, 2, 3]);
        assert_eq!(array.as_array_mut().shape(), [1, 2, 3]);
        assert_eq!(*array.get([0, 1, 2]).unwrap(), 0.0);
        assert_eq!(array.as_slice().unwrap().len(), 2 * 3);
        assert_eq!(array.as_slice_mut().unwrap().len(), 2 * 3);
        assert_eq!(*array.get_mut([0, 1, 2]).unwrap(), 0.0);
    });
}

#[test]
fn resize_using_exclusive_borrow() {
    Python::with_gil(|py| {
        let array = PyArray::<f64, _>::zeros_bound(py, 3, false);
        assert_eq!(array.shape(), [3]);

        let mut array = array.readwrite();
        assert_eq!(array.as_slice_mut().unwrap(), &[0.0; 3]);

        let mut array = array.resize(5).unwrap();
        assert_eq!(array.as_slice_mut().unwrap(), &[0.0; 5]);
    });
}

#[cfg(feature = "nalgebra")]
#[test]
fn matrix_from_numpy() {
    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];

        {
            let array = array.readonly();

            let matrix = array.as_matrix();
            assert_eq!(matrix, nalgebra::Matrix3::new(0, 1, 2, 3, 4, 5, 6, 7, 8));

            let matrix: nalgebra::MatrixView<
                '_,
                i32,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<1>,
            > = array.try_as_matrix().unwrap();
            assert_eq!(matrix, nalgebra::Matrix3::new(0, 1, 2, 3, 4, 5, 6, 7, 8));
        }

        {
            let array = array.readwrite();

            let matrix = array.as_matrix_mut();
            assert_eq!(matrix, nalgebra::Matrix3::new(0, 1, 2, 3, 4, 5, 6, 7, 8));

            let matrix: nalgebra::MatrixViewMut<
                '_,
                i32,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<1>,
            > = array.try_as_matrix_mut().unwrap();
            assert_eq!(matrix, nalgebra::Matrix3::new(0, 1, 2, 3, 4, 5, 6, 7, 8));
        }
    });

    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, 0, 1, 2];

        {
            let array = array.readonly();

            let matrix = array.as_matrix();
            assert_eq!(matrix, nalgebra::Matrix3x1::new(0, 1, 2));

            let matrix: nalgebra::MatrixView<'_, i32, nalgebra::Const<3>, nalgebra::Const<1>> =
                array.try_as_matrix().unwrap();
            assert_eq!(matrix, nalgebra::Matrix3x1::new(0, 1, 2));
        }

        {
            let array = array.readwrite();

            let matrix = array.as_matrix_mut();
            assert_eq!(matrix, nalgebra::Matrix3x1::new(0, 1, 2));

            let matrix: nalgebra::MatrixViewMut<'_, i32, nalgebra::Const<3>, nalgebra::Const<1>> =
                array.try_as_matrix_mut().unwrap();
            assert_eq!(matrix, nalgebra::Matrix3x1::new(0, 1, 2));
        }
    });

    Python::with_gil(|py| {
        let array = PyArray::<i32, _>::zeros_bound(py, (2, 2, 2), false);
        let array = array.readonly();

        let matrix: Option<nalgebra::DMatrixView<'_, i32, nalgebra::Dyn, nalgebra::Dyn>> =
            array.try_as_matrix();
        assert!(matrix.is_none());
    });

    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];
        let array = py
            .eval_bound(
                "a[::-1]",
                Some(&[("a", array)].into_py_dict_bound(py)),
                None,
            )
            .unwrap()
            .downcast_into::<PyArray2<i32>>()
            .unwrap();
        let array = array.readonly();

        let matrix: Option<nalgebra::DMatrixView<'_, i32, nalgebra::Dyn, nalgebra::Dyn>> =
            array.try_as_matrix();
        assert!(matrix.is_none());
    });

    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, [[0, 1], [2, 3]], [[4, 5], [6, 7]]];
        let array = py
            .eval_bound(
                "a[:,:,0]",
                Some(&[("a", &array)].into_py_dict_bound(py)),
                None,
            )
            .unwrap()
            .downcast_into::<PyArray2<i32>>()
            .unwrap();
        let array = array.readonly();

        let matrix: nalgebra::MatrixView<
            '_,
            i32,
            nalgebra::Const<2>,
            nalgebra::Const<2>,
            nalgebra::Dyn,
            nalgebra::Dyn,
        > = array.try_as_matrix().unwrap();
        assert_eq!(matrix, nalgebra::Matrix2::new(0, 2, 4, 6));
    });

    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, [[0, 1], [2, 3]], [[4, 5], [6, 7]]];
        let array = py
            .eval_bound(
                "a[:,:,0]",
                Some(&[("a", &array)].into_py_dict_bound(py)),
                None,
            )
            .unwrap()
            .downcast_into::<PyArray2<i32>>()
            .unwrap();
        let array = array.readonly();

        let matrix: nalgebra::MatrixView<
            '_,
            i32,
            nalgebra::Const<2>,
            nalgebra::Const<2>,
            nalgebra::Dyn,
            nalgebra::Dyn,
        > = array.try_as_matrix().unwrap();
        assert_eq!(matrix, nalgebra::Matrix2::new(0, 2, 4, 6));
    });

    Python::with_gil(|py| {
        let array = numpy::pyarray_bound![py, [0, 1, 2], [3, 4, 5], [6, 7, 8]];
        let array = array.readonly();

        let matrix: Option<
            nalgebra::MatrixView<
                '_,
                i32,
                nalgebra::Const<2>,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<1>,
            >,
        > = array.try_as_matrix();
        assert!(matrix.is_none());

        let matrix: Option<
            nalgebra::MatrixView<
                '_,
                i32,
                nalgebra::Const<3>,
                nalgebra::Const<2>,
                nalgebra::Const<3>,
                nalgebra::Const<1>,
            >,
        > = array.try_as_matrix();
        assert!(matrix.is_none());

        let matrix: Option<
            nalgebra::MatrixView<
                '_,
                i32,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<2>,
                nalgebra::Const<1>,
            >,
        > = array.try_as_matrix();
        assert!(matrix.is_none());

        let matrix: Option<
            nalgebra::MatrixView<
                '_,
                i32,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<3>,
                nalgebra::Const<2>,
            >,
        > = array.try_as_matrix();
        assert!(matrix.is_none());
    });
}

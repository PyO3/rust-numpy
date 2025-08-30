use std::mem::size_of;

#[cfg(feature = "half")]
use half::{bf16, f16};
use ndarray::{array, s, Array1, Dim};
use numpy::prelude::*;
use numpy::{
    dtype, get_array_module, npyffi::NPY_ORDER, pyarray, PyArray, PyArray1, PyArray2, PyArrayDescr,
    PyFixedString, PyFixedUnicode,
};
use pyo3::ffi::c_str;
use pyo3::{
    py_run, pyclass, pymethods,
    types::{IntoPyDict, PyAnyMethods, PyDict, PyList},
    Bound, Py, Python,
};

fn get_np_locals(py: Python<'_>) -> Bound<'_, PyDict> {
    [("np", get_array_module(py).unwrap())]
        .into_py_dict(py)
        .unwrap()
}

fn not_contiguous_array(py: Python<'_>) -> Bound<'_, PyArray1<i32>> {
    py.eval(
        c_str!("np.array([1, 2, 3, 4], dtype='int32')[::2]"),
        None,
        Some(&get_np_locals(py)),
    )
    .unwrap()
    .cast_into()
    .unwrap()
}

#[test]
fn new_c_order() {
    Python::attach(|py| {
        let dims = [3, 5];

        let arr = PyArray::<f64, _>::zeros(py, dims, false);

        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dims);

        let size = size_of::<f64>() as isize;
        assert!(arr.strides() == [dims[1] as isize * size, size]);

        assert_eq!(arr.shape(), &dims);
        assert_eq!(arr.len(), 3 * 5);

        assert!(arr.is_contiguous());
        assert!(arr.is_c_contiguous());
        assert!(!arr.is_fortran_contiguous());
    });
}

#[test]
fn new_fortran_order() {
    Python::attach(|py| {
        let dims = [3, 5];

        let arr = PyArray::<f64, _>::zeros(py, dims, true);

        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dims);

        let size = size_of::<f64>() as isize;
        assert!(arr.strides() == [size, dims[0] as isize * size]);

        assert_eq!(arr.shape(), &dims);
        assert_eq!(arr.len(), 3 * 5);

        assert!(arr.is_contiguous());
        assert!(!arr.is_c_contiguous());
        assert!(arr.is_fortran_contiguous());
    });
}

#[test]
fn tuple_as_dim() {
    Python::attach(|py| {
        let dims = (3, 5);

        let arr = PyArray::<f64, _>::zeros(py, dims, false);

        assert!(arr.ndim() == 2);
        assert!(arr.dims() == [3, 5]);
    });
}

#[test]
fn rank_zero_array_has_invalid_strides_dimensions() {
    Python::attach(|py| {
        let arr = PyArray::<f64, _>::zeros(py, (), false);

        assert_eq!(arr.ndim(), 0);
        assert_eq!(arr.strides(), &[] as &[isize]);
        assert_eq!(arr.shape(), &[] as &[usize]);

        assert_eq!(arr.len(), 1);
        assert!(!arr.is_empty());

        assert_eq!(arr.item(), 0.0);
    })
}

#[test]
fn zeros() {
    Python::attach(|py| {
        let dims = [3, 4];

        let arr = PyArray::<f64, _>::zeros(py, dims, false);

        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dims);

        let size = size_of::<f64>() as isize;
        assert!(arr.strides() == [dims[1] as isize * size, size]);

        let arr = PyArray::<f64, _>::zeros(py, dims, true);

        assert!(arr.ndim() == 2);
        assert!(arr.dims() == dims);

        let size = size_of::<f64>() as isize;
        assert!(arr.strides() == [size, dims[0] as isize * size]);
    });
}

#[test]
fn arange() {
    Python::attach(|py| {
        let arr = PyArray::<f64, _>::arange(py, 0.0, 1.0, 0.1);

        assert_eq!(arr.ndim(), 1);
        assert_eq!(arr.dims(), Dim([10]));
    });
}

#[test]
fn as_array() {
    Python::attach(|py| {
        let pyarr = PyArray::<f64, _>::zeros(py, [3, 2, 4], false).readonly();
        let arr = pyarr.as_array();

        assert_eq!(pyarr.shape(), arr.shape());
        assert_eq!(
            pyarr
                .strides()
                .iter()
                .map(|x| x / size_of::<f64>() as isize)
                .collect::<Vec<_>>(),
            arr.strides()
        );

        let not_contiguous = not_contiguous_array(py).readonly();
        assert_eq!(not_contiguous.as_array(), array![1, 3]);
    });
}

#[test]
fn as_raw_array() {
    Python::attach(|py| {
        let not_contiguous = not_contiguous_array(py);

        let raw_array_view = not_contiguous.as_raw_array();
        assert_eq!(unsafe { raw_array_view.deref_into_view()[0] }, 1);

        let raw_array_view_mut = not_contiguous.as_raw_array_mut();
        assert_eq!(unsafe { raw_array_view_mut.deref_into_view_mut()[1] }, 3);
    });
}

#[test]
fn as_slice() {
    Python::attach(|py| {
        let arr = PyArray::<i32, _>::zeros(py, [3, 2, 4], false);
        assert_eq!(arr.readonly().as_slice().unwrap().len(), 3 * 2 * 4);

        let not_contiguous = not_contiguous_array(py);
        let err = not_contiguous.readonly().as_slice().unwrap_err();
        assert_eq!(err.to_string(), "The given array is not contiguous");
    });
}

#[test]
fn is_instance() {
    Python::attach(|py| {
        let arr = PyArray2::<f64>::zeros(py, [3, 5], false);

        assert!(arr.is_instance_of::<PyArray2<f64>>());
        assert!(!arr.is_instance_of::<PyList>());
    });
}

#[test]
fn from_vec2() {
    Python::attach(|py| {
        let pyarray = PyArray::from_vec2(py, &[vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

        assert_eq!(pyarray.readonly().as_array(), array![[1, 2, 3], [4, 5, 6]]);
    });
}

#[test]
fn from_vec2_ragged() {
    Python::attach(|py| {
        let pyarray = PyArray::from_vec2(py, &[vec![1, 2, 3], vec![4, 5]]);

        let err = pyarray.unwrap_err();
        assert_eq!(err.to_string(), "invalid length: 2, but expected 3");
    });
}

#[test]
fn from_vec3() {
    Python::attach(|py| {
        let pyarray = PyArray::from_vec3(
            py,
            &[
                vec![vec![1, 2], vec![3, 4]],
                vec![vec![5, 6], vec![7, 8]],
                vec![vec![9, 10], vec![11, 12]],
            ],
        )
        .unwrap();

        assert_eq!(
            pyarray.readonly().as_array(),
            array![[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        );
    });
}

#[test]
fn from_vec3_ragged() {
    Python::attach(|py| {
        let pyarray = PyArray::from_vec3(
            py,
            &[
                vec![vec![1, 2], vec![3, 4]],
                vec![vec![5, 6], vec![7, 8]],
                vec![vec![9, 10], vec![11]],
            ],
        );

        let err = pyarray.unwrap_err();
        assert_eq!(err.to_string(), "invalid length: 1, but expected 2");

        let pyarray = PyArray::from_vec3(
            py,
            &[
                vec![vec![1, 2], vec![3, 4]],
                vec![vec![5, 6], vec![7, 8]],
                vec![vec![9, 10]],
            ],
        );

        let err = pyarray.unwrap_err();
        assert_eq!(err.to_string(), "invalid length: 1, but expected 2");
    });
}

#[test]
fn array_cast() {
    Python::attach(|py| {
        let arr_f64 = pyarray![py, [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]];
        let arr_i32 = arr_f64.cast_array::<i32>(false).unwrap();

        assert_eq!(arr_i32.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
    });
}

#[test]
fn handle_negative_strides() {
    Python::attach(|py| {
        let arr = array![[2, 3], [4, 5u32]];
        let pyarr = arr.to_pyarray(py);

        let neg_str_pyarr = py
            .eval(
                c_str!("a[::-1]"),
                Some(&[("a", pyarr)].into_py_dict(py).unwrap()),
                None,
            )
            .unwrap()
            .cast_into::<PyArray2<u32>>()
            .unwrap();

        assert_eq!(
            neg_str_pyarr.readonly().as_array(),
            arr.slice(s![..;-1, ..])
        );
    });
}

#[test]
fn dtype_via_python_attribute() {
    Python::attach(|py| {
        let arr = array![[2, 3], [4, 5u32]];
        let pyarr = arr.to_pyarray(py);

        let dt = py
            .eval(
                c_str!("a.dtype"),
                Some(&[("a", pyarr)].into_py_dict(py).unwrap()),
                None,
            )
            .unwrap()
            .cast_into::<PyArrayDescr>()
            .unwrap();

        assert!(dt.is_equiv_to(&dtype::<u32>(py)));
    });
}

#[pyclass]
struct Owner {
    array: Array1<f64>,
}

#[pymethods]
impl Owner {
    #[getter]
    fn array(this: Bound<'_, Self>) -> Bound<'_, PyArray1<f64>> {
        let array = &this.borrow().array;

        unsafe { PyArray1::borrow_from_array(array, this.into_any()) }
    }
}

#[test]
fn borrow_from_array_works() {
    let array = Python::attach(|py| {
        let owner = Py::new(
            py,
            Owner {
                array: Array1::linspace(0., 1., 10),
            },
        )
        .unwrap();

        owner.getattr(py, "array").unwrap()
    });

    Python::attach(|py| {
        py_run!(py, array, "assert array.shape == (10,)");
    });
}

#[test]
fn casting_works() {
    Python::attach(|py| {
        let ob = PyArray::from_slice(py, &[1_i32, 2, 3]).into_any();

        assert!(ob.cast::<PyArray1<i32>>().is_ok());
    });
}

#[test]
fn casting_respects_element_type() {
    Python::attach(|py| {
        let ob = PyArray::from_slice(py, &[1_i32, 2, 3]).into_any();

        assert!(ob.cast::<PyArray1<f64>>().is_err());
    });
}

#[test]
fn casting_respects_dimensionality() {
    Python::attach(|py| {
        let ob = PyArray::from_slice(py, &[1_i32, 2, 3]).into_any();

        assert!(ob.cast::<PyArray2<i32>>().is_err());
    });
}

#[test]
fn unbind_works() {
    let arr: Py<PyArray1<_>> = Python::attach(|py| {
        let arr = PyArray::from_slice(py, &[1_i32, 2, 3]);

        arr.unbind()
    });

    Python::attach(|py| {
        let arr = arr.bind(py);

        assert_eq!(arr.readonly().as_slice().unwrap(), &[1, 2, 3]);
    });
}

#[test]
fn copy_to_works() {
    Python::attach(|py| {
        let arr1 = PyArray::arange(py, 2.0, 5.0, 1.0);
        let arr2 = unsafe { PyArray::<i64, _>::new(py, [3], false) };

        arr1.copy_to(&arr2).unwrap();

        assert_eq!(arr2.readonly().as_slice().unwrap(), &[2, 3, 4]);
    });
}

#[test]
fn get_works() {
    Python::attach(|py| {
        let array = pyarray![py, [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]];

        unsafe {
            assert_eq!(array.get([0, 0, 0]), Some(&1));
            assert_eq!(array.get([2, 1, 1]), Some(&12));
            assert_eq!(array.get([0, 0, 2]), None);
            assert_eq!(array.get([0, 2, 0]), None);
            assert_eq!(array.get([3, 0, 0]), None);

            assert_eq!(*array.uget([1, 0, 0]), 5);
            assert_eq!(*array.uget_mut([0, 1, 0]), 3);
            assert_eq!(*array.uget_raw([0, 0, 1]), 2);
        }
    });
}

#[test]
fn permute_and_transpose() {
    Python::attach(|py| {
        let array = array![[0, 1, 2], [3, 4, 5]].into_pyarray(py);

        let permuted = array.permute(Some([1, 0])).unwrap();
        assert_eq!(
            permuted.readonly().as_array(),
            array![[0, 3], [1, 4], [2, 5]]
        );

        let permuted = array.permute::<()>(None).unwrap();
        assert_eq!(
            permuted.readonly().as_array(),
            array![[0, 3], [1, 4], [2, 5]]
        );

        let transposed = array.transpose().unwrap();
        assert_eq!(
            transposed.readonly().as_array(),
            array![[0, 3], [1, 4], [2, 5]]
        );

        let array = pyarray![py, [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]];

        let permuted = array.permute(Some([0, 2, 1])).unwrap();
        assert_eq!(
            permuted.readonly().as_array(),
            array![[[1, 3], [2, 4]], [[5, 7], [6, 8]], [[9, 11], [10, 12]]]
        );
    });
}

#[test]
fn reshape() {
    Python::attach(|py| {
        let array = PyArray::from_iter(py, 0..9)
            .reshape_with_order([3, 3], NPY_ORDER::NPY_FORTRANORDER)
            .unwrap();

        assert_eq!(
            array.readonly().as_array(),
            array![[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        );
        assert!(array.is_fortran_contiguous());

        assert!(array.reshape([5]).is_err());
    });
}

#[cfg(feature = "half")]
#[test]
fn half_f16_works() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        let locals = [("np", &np)].into_py_dict(py).unwrap();

        let array = py
            .eval(
                c_str!("np.array([[1, 2], [3, 4]], dtype='float16')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast_into::<PyArray2<f16>>()
            .unwrap();

        assert_eq!(
            array.readonly().as_array(),
            array![
                [f16::from_f32(1.0), f16::from_f32(2.0)],
                [f16::from_f32(3.0), f16::from_f32(4.0)]
            ]
        );

        array
            .readwrite()
            .as_array_mut()
            .map_inplace(|value| *value *= f16::from_f32(2.0));

        py_run!(
            py,
            array np,
            "assert np.all(array == np.array([[2, 4], [6, 8]], dtype='float16'))"
        );
    });
}

#[cfg(feature = "half")]
#[test]
fn half_bf16_works() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        // NumPy itself does not provide a `bfloat16` dtype itself,
        // so we import ml_dtypes which does register such a dtype.
        let mldt = py
            .eval(c_str!("__import__('ml_dtypes')"), None, None)
            .unwrap();
        let locals = [("np", &np), ("mldt", &mldt)].into_py_dict(py).unwrap();

        let array = py
            .eval(
                c_str!("np.array([[1, 2], [3, 4]], dtype='bfloat16')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast_into::<PyArray2<bf16>>()
            .unwrap();

        assert_eq!(
            array.readonly().as_array(),
            array![
                [bf16::from_f32(1.0), bf16::from_f32(2.0)],
                [bf16::from_f32(3.0), bf16::from_f32(4.0)]
            ]
        );

        array
            .readwrite()
            .as_array_mut()
            .map_inplace(|value| *value *= bf16::from_f32(2.0));

        py_run!(
            py,
            array np,
            "assert np.all(array == np.array([[2, 4], [6, 8]], dtype='bfloat16'))"
        );
    });
}

#[test]
fn ascii_strings_with_explicit_dtype_works() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        let locals = [("np", &np)].into_py_dict(py).unwrap();

        let array = py
            .eval(
                c_str!("np.array([b'foo', b'bar', b'foobar'], dtype='S6')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast_into::<PyArray1<PyFixedString<6>>>()
            .unwrap();

        {
            let array = array.readonly();
            let array = array.as_array();

            assert_eq!(array[0].0, [b'f', b'o', b'o', 0, 0, 0]);
            assert_eq!(array[1].0, [b'b', b'a', b'r', 0, 0, 0]);
            assert_eq!(array[2].0, [b'f', b'o', b'o', b'b', b'a', b'r']);
        }

        {
            let mut array = array.readwrite();
            let mut array = array.as_array_mut();

            array[2].0[5] = b'z';
        }

        py_run!(py, array np, "assert array[2] == b'foobaz'");
    });
}

#[test]
fn unicode_strings_with_explicit_dtype_works() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        let locals = [("np", &np)].into_py_dict(py).unwrap();

        let array = py
            .eval(
                c_str!("np.array(['foo', 'bar', 'foobar'], dtype='U6')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast_into::<PyArray1<PyFixedUnicode<6>>>()
            .unwrap();

        {
            let array = array.readonly();
            let array = array.as_array();

            assert_eq!(array[0].0, [b'f' as _, b'o' as _, b'o' as _, 0, 0, 0]);
            assert_eq!(array[1].0, [b'b' as _, b'a' as _, b'r' as _, 0, 0, 0]);
            assert_eq!(
                array[2].0,
                [
                    b'f' as u32,
                    b'o' as _,
                    b'o' as _,
                    b'b' as _,
                    b'a' as _,
                    b'r' as _
                ]
            );
        }

        {
            let mut array = array.readwrite();
            let mut array = array.as_array_mut();

            array[2].0[5] = b'z' as _;
        }

        py_run!(py, array np, "assert array[2] == 'foobaz'");
    });
}

#[test]
fn ascii_strings_ignore_byteorder() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        let locals = [("np", &np)].into_py_dict(py).unwrap();

        let native_endian_works = py
            .eval(
                c_str!("np.array([b'foo', b'bar'], dtype='=S3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedString<3>>>()
            .is_ok();

        let little_endian_works = py
            .eval(
                c_str!("np.array(['bfoo', b'bar'], dtype='<S3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedString<3>>>()
            .is_ok();

        let big_endian_works = py
            .eval(
                c_str!("np.array([b'foo', b'bar'], dtype='>S3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedString<3>>>()
            .is_ok();

        match (native_endian_works, little_endian_works, big_endian_works) {
            (true, true, true) => (),
            _ => panic!("All byteorders should work",),
        }
    });
}

#[test]
fn unicode_strings_respect_byteorder() {
    Python::attach(|py| {
        let np = py.eval(c_str!("__import__('numpy')"), None, None).unwrap();
        let locals = [("np", &np)].into_py_dict(py).unwrap();

        let native_endian_works = py
            .eval(
                c_str!("np.array(['foo', 'bar'], dtype='=U3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedUnicode<3>>>()
            .is_ok();

        let little_endian_works = py
            .eval(
                c_str!("np.array(['foo', 'bar'], dtype='<U3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedUnicode<3>>>()
            .is_ok();

        let big_endian_works = py
            .eval(
                c_str!("np.array(['foo', 'bar'], dtype='>U3')"),
                None,
                Some(&locals),
            )
            .unwrap()
            .cast::<PyArray1<PyFixedUnicode<3>>>()
            .is_ok();

        match (native_endian_works, little_endian_works, big_endian_works) {
            (true, true, false) | (true, false, true) => (),
            _ => panic!("Only native byteorder should work"),
        }
    });
}

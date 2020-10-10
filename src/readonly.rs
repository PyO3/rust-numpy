//! Readonly arrays
use crate::npyffi::NPY_ARRAY_WRITEABLE;
use crate::{Element, NotContiguousError, NpyIndex, PyArray};
use ndarray::{ArrayView, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use pyo3::{prelude::*, types::PyAny, AsPyPointer};

/// Readonly reference of [`PyArray`](../array/struct.PyArray.html).
///
/// This struct ensures that the internal array is not writeable while holding `PyReadonlyArray`.
/// We use a simple trick for this: modifying the internal flag of the array when creating
/// `PyReadonlyArray` and recover the original flag when it drops.
///
/// So, importantly, it does not recover the original flag when it does not drop
/// (e.g.,  by the use of `IntoPy::intopy` or `std::mem::forget`)
/// and then the internal array remains readonly.
///
/// # Example
/// In this example, we get a 'temporal' readonly array and the internal array
/// becomes writeble again after it drops.
/// ```
/// use numpy::{PyArray, npyffi::NPY_ARRAY_WRITEABLE};
/// pyo3::Python::with_gil(|py| {
///     let py_array = PyArray::arange(py, 0, 4, 1).reshape([2, 2]).unwrap();
///     {
///        let readonly = py_array.readonly();
///        // The internal array is not writeable now.
///        pyo3::py_run!(py, py_array, "assert not py_array.flags['WRITEABLE']");
///     }
///     // After the `readonly` drops, the internal array gets writeable again.
///     pyo3::py_run!(py, py_array, "assert py_array.flags['WRITEABLE']");
/// });
/// ```
/// However, if we convert the `PyReadonlyArray` directly into `PyObject`,
/// the internal array remains readonly.
/// ```
/// use numpy::{PyArray, npyffi::NPY_ARRAY_WRITEABLE};
/// use pyo3::{IntoPy, PyObject, Python};
/// pyo3::Python::with_gil(|py| {
///     let py_array = PyArray::arange(py, 0, 4, 1).reshape([2, 2]).unwrap();
///     let obj: PyObject = {
///        let readonly = py_array.readonly();
///        // The internal array is not writeable now.
///        pyo3::py_run!(py, py_array, "assert not py_array.flags['WRITEABLE']");
///        readonly.into_py(py)
///     };
///     // The internal array remains readonly.
///     pyo3::py_run!(py, py_array, "assert py_array.flags['WRITEABLE']");
/// });
/// ```
pub struct PyReadonlyArray<'py, T, D> {
    array: &'py PyArray<T, D>,
    was_writeable: bool,
}

impl<'py, T: Element, D: Dimension> PyReadonlyArray<'py, T, D> {
    /// Returns the immutable view of the internal data of `PyArray` as slice.
    ///
    /// Returns `ErrorKind::NotContiguous` if the internal array is not contiguous.
    /// # Example
    /// ```
    /// use numpy::{PyArray, PyArray1};
    /// use pyo3::types::IntoPyDict;
    /// pyo3::Python::with_gil(|py| {
    ///     let py_array = PyArray::arange(py, 0, 4, 1).reshape([2, 2]).unwrap();
    ///     let readonly = py_array.readonly();
    ///     assert_eq!(readonly.as_slice().unwrap(), &[0, 1, 2, 3]);
    ///     let locals = [("np", numpy::get_array_module(py).unwrap())].into_py_dict(py);
    ///     let not_contiguous: &PyArray1<i32> = py
    ///         .eval("np.arange(10)[::2]", Some(locals), None)
    ///         .unwrap()
    ///         .downcast()
    ///         .unwrap();
    ///     assert!(not_contiguous.readonly().as_slice().is_err());
    /// });
    /// ```
    pub fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        unsafe { self.array.as_slice() }
    }

    /// Get the immutable view of the internal data of `PyArray`, as
    /// [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// pyo3::Python::with_gil(|py| {
    ///     let array = PyArray::arange(py, 0, 4, 1).reshape([2, 2]).unwrap();
    ///     let readonly = array.readonly();
    ///     assert_eq!(readonly.as_array(), array![[0, 1], [2, 3]]);
    /// });
    /// ```
    pub fn as_array(&self) -> ArrayView<'_, T, D> {
        unsafe { self.array.as_array() }
    }

    /// Get an immutable reference of the specified element, with checking the passed index is valid.
    ///
    /// See [NpyIndex](../convert/trait.NpyIndex.html) for what types you can use as index.
    ///
    /// If you pass an invalid index to this function, it returns `None`.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// pyo3::Python::with_gil(|py| {
    ///     let arr = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap().readonly();
    ///     assert_eq!(*arr.get([1, 0, 3]).unwrap(), 11);
    ///     assert!(arr.get([2, 0, 3]).is_none());
    /// });
    /// ```
    ///
    /// For fixed dimension arrays, passing an index with invalid dimension causes compile error.
    /// ```compile_fail
    /// use numpy::PyArray;
    /// pyo3::Python::with_gil(|py| {
    ///     let arr = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap().readonly();
    ///     let a = arr.get([1, 2]); // Compile Error!
    /// });
    /// ```
    ///
    /// However, for dinamic arrays, we cannot raise a compile error and just returns `None`.
    /// ```
    /// use numpy::PyArray;
    /// pyo3::Python::with_gil(|py| {
    ///     let arr = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap().readonly();
    ///     let arr = arr.to_dyn().readonly();
    ///     assert!(arr.get([1, 2].as_ref()).is_none());
    /// });
    /// ```
    #[inline(always)]
    pub fn get(&self, index: impl NpyIndex<Dim = D>) -> Option<&T> {
        unsafe { self.array.get(index) }
    }

    /// Iterates all elements of this array.
    /// See [NpySingleIter](../npyiter/struct.NpySingleIter.html) for more.
    pub fn iter(self) -> PyResult<crate::NpySingleIter<'py, T, crate::npyiter::Readonly>> {
        crate::NpySingleIterBuilder::readonly(self).build()
    }

    pub(crate) fn destruct(self) -> (&'py PyArray<T, D>, bool) {
        let PyReadonlyArray {
            array,
            was_writeable,
        } = self;
        (array, was_writeable)
    }
}

/// One-dimensional readonly array.
pub type PyReadonlyArray1<'py, T> = PyReadonlyArray<'py, T, Ix1>;
/// Two-dimensional readonly array.
pub type PyReadonlyArray2<'py, T> = PyReadonlyArray<'py, T, Ix2>;
/// Three-dimensional readonly array.
pub type PyReadonlyArray3<'py, T> = PyReadonlyArray<'py, T, Ix3>;
/// Four-dimensional readonly array.
pub type PyReadonlyArray4<'py, T> = PyReadonlyArray<'py, T, Ix4>;
/// Five-dimensional readonly array.
pub type PyReadonlyArray5<'py, T> = PyReadonlyArray<'py, T, Ix5>;
/// Six-dimensional readonly array.
pub type PyReadonlyArray6<'py, T> = PyReadonlyArray<'py, T, Ix6>;
/// Dynamic-dimensional readonly array.
pub type PyReadonlyArrayDyn<'py, T> = PyReadonlyArray<'py, T, IxDyn>;

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for PyReadonlyArray<'py, T, D> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        let array: &PyArray<T, D> = obj.extract()?;
        Ok(PyReadonlyArray::from(array))
    }
}

impl<'py, T, D> IntoPy<PyObject> for PyReadonlyArray<'py, T, D> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let PyReadonlyArray { array, .. } = self;
        unsafe { PyObject::from_borrowed_ptr(py, array.as_ptr()) }
    }
}

impl<'py, T, D> From<&'py PyArray<T, D>> for PyReadonlyArray<'py, T, D> {
    fn from(array: &'py PyArray<T, D>) -> PyReadonlyArray<'py, T, D> {
        let flag = array.get_flag();
        let writeable = flag & NPY_ARRAY_WRITEABLE != 0;
        if writeable {
            unsafe {
                (*array.as_array_ptr()).flags &= !NPY_ARRAY_WRITEABLE;
            }
        }
        Self {
            array,
            was_writeable: writeable,
        }
    }
}

impl<'py, T, D> Drop for PyReadonlyArray<'py, T, D> {
    fn drop(&mut self) {
        if self.was_writeable {
            unsafe {
                (*self.array.as_array_ptr()).flags |= NPY_ARRAY_WRITEABLE;
            }
        }
    }
}

impl<'py, T, D> AsRef<PyArray<T, D>> for PyReadonlyArray<'py, T, D> {
    fn as_ref(&self) -> &PyArray<T, D> {
        self.array
    }
}

impl<'py, T, D> std::ops::Deref for PyReadonlyArray<'py, T, D> {
    type Target = PyArray<T, D>;
    fn deref(&self) -> &PyArray<T, D> {
        self.array
    }
}

//! Defines conversion traits between rust types and numpy data types.

use ndarray::{ArrayBase, Data, Dimension};
use pyo3::Python;

use std::os::raw::c_int;

use super::*;

/// Covversion trait from rust types to `PyArray`.
///
/// This trait takes `&self`, which means **it alocates in Python heap and then copies
/// elements there**.
/// # Example
/// ```
/// # extern crate pyo3; extern crate numpy; fn main() {
/// use numpy::{PyArray, ToPyArray};
/// let gil = pyo3::Python::acquire_gil();
/// let py_array = vec![1, 2, 3].to_pyarray(gil.python());
/// assert_eq!(py_array.as_slice().unwrap(), &[1, 2, 3]);
/// # }
/// ```
pub trait ToPyArray {
    type Item: TypeNum;
    fn to_pyarray<'py>(&self, Python<'py>) -> &'py PyArray<Self::Item>;
}

impl<T: TypeNum> ToPyArray for [T] {
    type Item = T;
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item> {
        PyArray::from_slice(py, self)
    }
}

impl<S, D, A> ToPyArray for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: TypeNum,
{
    type Item = A;
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item> {
        PyArray::from_ndarray(py, self)
    }
}

/// Utility trait to specify the dimention of array
pub trait ToNpyDims: Dimension {
    fn ndim_cint(&self) -> c_int {
        self.ndim() as c_int
    }
    fn as_dims_ptr(&self) -> *mut npy_intp {
        self.slice().as_ptr() as *mut npy_intp
    }
    fn to_npy_dims(&self) -> npyffi::PyArray_Dims {
        npyffi::PyArray_Dims {
            ptr: self.as_dims_ptr(),
            len: self.ndim_cint(),
        }
    }
    fn __private__(&self) -> PrivateMarker;
}

impl<T: Dimension> ToNpyDims for T {
    fn __private__(&self) -> PrivateMarker {
        PrivateMarker
    }
}

#[doc(hidden)]
pub struct PrivateMarker;

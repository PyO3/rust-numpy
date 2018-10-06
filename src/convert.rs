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
pub trait ToNpyDims {
    fn dims_len(&self) -> c_int;
    fn dims_ptr(&self) -> *mut npy_intp;
    fn dims_ref(&self) -> &[usize];
    fn to_npy_dims(&self) -> npyffi::PyArray_Dims {
        npyffi::PyArray_Dims {
            ptr: self.dims_ptr(),
            len: self.dims_len(),
        }
    }
}

macro_rules! array_dim_impls {
    ($($N: expr)+) => {
        $(
            impl ToNpyDims for [usize; $N] {
                fn dims_len(&self) -> c_int {
                    $N as c_int
                }
                fn dims_ptr(&self) -> *mut npy_intp {
                    self.as_ptr() as *mut npy_intp
                }
                fn dims_ref(&self) -> &[usize] {
                    self
                }
            }
        )+
    }
}

array_dim_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16
}

impl<'a> ToNpyDims for &'a [usize] {
    fn dims_len(&self) -> c_int {
        self.len() as c_int
    }
    fn dims_ptr(&self) -> *mut npy_intp {
        self.as_ptr() as *mut npy_intp
    }
    fn dims_ref(&self) -> &[usize] {
        *self
    }
}

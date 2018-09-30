//! Defines conversion trait between rust types and numpy data types.

use ndarray::*;
use pyo3::Python;

use std::iter::Iterator;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::null_mut;

use super::*;

/// Covversion trait from rust types to `PyArray`.
///
/// # Example
/// ```
/// # extern crate pyo3; extern crate numpy; fn main() {
/// use numpy::{PyArray, IntoPyArray};
/// let gil = pyo3::Python::acquire_gil();
/// let py_array = vec![1, 2, 3].into_pyarray(gil.python());
/// assert_eq!(py_array.as_slice().unwrap(), &[1, 2, 3]);
/// # }
/// ```
pub trait IntoPyArray {
    type Item: TypeNum;
    fn into_pyarray(self, Python) -> &PyArray<Self::Item>;
}

impl<T: TypeNum> IntoPyArray for Box<[T]> {
    type Item = T;
    fn into_pyarray(self, py: Python) -> &PyArray<Self::Item> {
        let dims = [self.len()];
        let ptr = Box::into_raw(self);
        unsafe { PyArray::new_(py, &dims, null_mut(), ptr as *mut c_void) }
    }
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    type Item = T;
    fn into_pyarray(self, py: Python) -> &PyArray<Self::Item> {
        let dims = [self.len()];
        unsafe { PyArray::new_(py, &dims, null_mut(), into_raw(self)) }
    }
}

impl<A: TypeNum, D: Dimension> IntoPyArray for Array<A, D> {
    type Item = A;
    fn into_pyarray(self, py: Python) -> &PyArray<Self::Item> {
        let dims: Vec<_> = self.shape().iter().cloned().collect();
        let mut strides: Vec<_> = self
            .strides()
            .into_iter()
            .map(|n| n * size_of::<A>() as npy_intp)
            .collect();
        unsafe {
            let data = into_raw(self.into_raw_vec());
            PyArray::new_(py, &dims, strides.as_mut_ptr(), data)
        }
    }
}

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            impl<T: TypeNum> IntoPyArray for [T; $N] {
                type Item = T;
                fn into_pyarray(self, py: Python) -> &PyArray<T> {
                    let dims = [$N];
                    let ptr = Box::into_raw(Box::new(self));
                    unsafe {
                        PyArray::new_(py, &dims, null_mut(), ptr as *mut c_void)
                    }
                }
            }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

pub(crate) unsafe fn into_raw<T>(x: Vec<T>) -> *mut c_void {
    let ptr = Box::into_raw(x.into_boxed_slice());
    ptr as *mut c_void
}

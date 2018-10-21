//! Defines conversion traits between rust types and numpy data types.

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix1, OwnedRepr};
use pyo3::Python;

use std::mem;
use std::os::raw::c_int;
use std::ptr;

use super::*;
use npyffi::npy_intp;

/// Covnersion trait from some rust types to `PyArray`.
///
/// This trait takes `self`, which means **it holds a pointer to Rust heap, until `resize` or other
/// destructive method is called**.
/// # Example
/// ```
/// # extern crate pyo3; extern crate numpy; fn main() {
/// use numpy::{PyArray, ToPyArray};
/// let gil = pyo3::Python::acquire_gil();
/// let py_array = vec![1, 2, 3].to_pyarray(gil.python());
/// assert_eq!(py_array.as_slice().unwrap(), &[1, 2, 3]);
/// # }
/// ```
pub trait IntoPyArray {
    type Item: TypeNum;
    type Dim: Dimension;
    fn into_pyarray<'py>(self, Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
}

impl<T: TypeNum> IntoPyArray for Box<[T]> {
    type Item = T;
    type Dim = Ix1;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let len = self.len();
        unsafe { PyArray::from_boxed_slice(py, [len], ptr::null_mut(), self) }
    }
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    type Item = T;
    type Dim = Ix1;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        self.into_boxed_slice().into_pyarray(py)
    }
}

impl<A, D> IntoPyArray for ArrayBase<OwnedRepr<A>, D>
where
    A: TypeNum,
    D: Dimension,
{
    type Item = A;
    type Dim = D;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let mut strides = npy_strides(&self);
        let dim = self.raw_dim();
        let boxed = self.into_raw_vec().into_boxed_slice();
        unsafe { PyArray::from_boxed_slice(py, dim, strides.as_mut_ptr() as *mut npy_intp, boxed) }
    }
}

/// Conversion trait from rust types to `PyArray`.
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
    type Dim: Dimension;
    fn to_pyarray<'py>(&self, Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
}

impl<T: TypeNum> ToPyArray for [T] {
    type Item = T;
    type Dim = Ix1;
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
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
    type Dim = D;
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let len = self.len();
        let mut strides = npy_strides(self);
        unsafe {
            let array = PyArray::new_(py, self.raw_dim(), strides.as_mut_ptr() as *mut npy_intp, 0);
            array.copy_ptr(self.as_ptr(), len);
            array
        }
    }
}

fn npy_strides<S, D, A>(array: &ArrayBase<S, D>) -> Vec<npyffi::npy_intp>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: TypeNum,
{
    array
        .strides()
        .into_iter()
        .map(|n| n * mem::size_of::<A>() as npyffi::npy_intp)
        .collect()
}

/// Utility trait to specify the dimention of array
pub trait ToNpyDims: Dimension {
    fn ndim_cint(&self) -> c_int {
        self.ndim() as c_int
    }
    fn as_dims_ptr(&self) -> *mut npyffi::npy_intp {
        self.slice().as_ptr() as *mut npyffi::npy_intp
    }
    fn to_npy_dims(&self) -> npyffi::PyArray_Dims {
        npyffi::PyArray_Dims {
            ptr: self.as_dims_ptr(),
            len: self.ndim_cint(),
        }
    }
    fn __private__(&self) -> PrivateMarker;
}

impl<D: Dimension> ToNpyDims for D {
    fn __private__(&self) -> PrivateMarker {
        PrivateMarker
    }
}

/// Types that can be used to index an array.
///
/// See
/// [IntoDimension](https://docs.rs/ndarray/0.12/ndarray/dimension/conversion/trait.IntoDimension.html)
/// for what types you can use as `NpyIndex`.
///
/// But basically, you can use
/// - Tuple
/// - Fixed sized array
/// - Slice
// Since Numpy's strides is byte offset, we can't use ndarray::NdIndex directly here.
pub trait NpyIndex: IntoDimension {
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize>;
    fn get_unchecked<T>(self, strides: &[isize]) -> isize;
    fn __private__(self) -> PrivateMarker;
}

impl<D: IntoDimension> NpyIndex for D {
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize> {
        let indices_ = self.into_dimension();
        let indices = indices_.slice();
        if indices.len() != dims.len() {
            return None;
        }
        if indices.into_iter().zip(dims).any(|(i, d)| i >= d) {
            return None;
        }
        Some(get_unchecked_impl(
            indices,
            strides,
            mem::size_of::<T>() as isize,
        ))
    }
    fn get_unchecked<T>(self, strides: &[isize]) -> isize {
        let indices_ = self.into_dimension();
        let indices = indices_.slice();
        get_unchecked_impl(indices, strides, mem::size_of::<T>() as isize)
    }
    fn __private__(self) -> PrivateMarker {
        PrivateMarker
    }
}

fn get_unchecked_impl(indices: &[usize], strides: &[isize], size: isize) -> isize {
    indices
        .iter()
        .zip(strides)
        .map(|(&i, stride)| stride * i as isize / size)
        .sum()
}

#[doc(hidden)]
pub struct PrivateMarker;

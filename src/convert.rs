//! Defines conversion traits between rust types and numpy data types.

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix1, OwnedRepr};
use pyo3::Python;

use std::{mem, os::raw::c_int};

use crate::{
    npyffi::{self, npy_intp},
    Element, PyArray,
};

/// Covnersion trait from some rust types to `PyArray`.
///
/// This trait takes `self`, which means **it holds a pointer to Rust heap, until `resize` or other
/// destructive method is called**.
///
/// In addition, if you construct `PyArray` via this method,
/// **you cannot use some destructive methods like `resize`.**
///
/// # Example
/// ```
/// use numpy::{PyArray, IntoPyArray};
/// pyo3::Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].into_pyarray(py);
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
///     assert!(py_array.resize(100).is_err()); // You can't resize owned-by-rust array.
/// });
/// ```
pub trait IntoPyArray {
    type Item: Element;
    type Dim: Dimension;
    fn into_pyarray<'py>(self, _: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
}

impl<T: Element> IntoPyArray for Box<[T]> {
    type Item = T;
    type Dim = Ix1;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let len = self.len();
        let strides = [mem::size_of::<T>() as npy_intp];
        unsafe { PyArray::from_boxed_slice(py, [len], strides.as_ptr(), self) }
    }
}

impl<T: Element> IntoPyArray for Vec<T> {
    type Item = T;
    type Dim = Ix1;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        self.into_boxed_slice().into_pyarray(py)
    }
}

impl<A, D> IntoPyArray for ArrayBase<OwnedRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    type Item = A;
    type Dim = D;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let strides = self.npy_strides();
        let dim = self.raw_dim();
        let boxed = self.into_raw_vec().into_boxed_slice();
        unsafe { PyArray::from_boxed_slice(py, dim, strides.as_ptr(), boxed) }
    }
}

/// Conversion trait from rust types to `PyArray`.
///
/// This trait takes `&self`, which means **it alocates in Python heap and then copies
/// elements there**.
/// # Example
/// ```
/// use numpy::{PyArray, ToPyArray};
/// pyo3::Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].to_pyarray(py);
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
/// });
/// ```
///
/// This method converts a not-contiguous array to C-order contiguous array.
/// # Example
/// ```
/// use numpy::{PyArray, ToPyArray};
/// use ndarray::{arr3, s};
/// pyo3::Python::with_gil(|py| {
///     let a = arr3(&[[[ 1,  2,  3], [ 4,  5,  6]],
///                    [[ 7,  8,  9], [10, 11, 12]]]);
///     let slice = a.slice(s![.., 0..1, ..]);
///     let sliced = arr3(&[[[ 1,  2,  3]],
///                         [[ 7,  8,  9]]]);
///     let py_slice = slice.to_pyarray(py);
///     assert_eq!(py_slice.readonly().as_array(), sliced);
///     pyo3::py_run!(py, py_slice, "assert py_slice.flags['C_CONTIGUOUS']");
/// });
/// ```
pub trait ToPyArray {
    type Item: Element;
    type Dim: Dimension;
    fn to_pyarray<'py>(&self, _: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
}

impl<T: Element> ToPyArray for [T] {
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
    A: Element,
{
    type Item = A;
    type Dim = D;
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let len = self.len();
        if let Some(order) = self.order() {
            // if the array is contiguous, copy it by `copy_ptr`.
            let strides = self.npy_strides();
            unsafe {
                let array = PyArray::new_(py, self.raw_dim(), strides.as_ptr(), order.to_flag());
                array.copy_ptr(self.as_ptr(), len);
                array
            }
        } else {
            // if the array is not contiguous, copy all elements by `ArrayBase::iter`.
            let dim = self.raw_dim();
            let strides = NpyStrides::from_dim(&dim, mem::size_of::<A>());
            unsafe {
                let array = PyArray::<A, _>::new_(py, dim, strides.as_ptr(), 0);
                let data_ptr = array.data();
                for (i, item) in self.iter().enumerate() {
                    data_ptr.add(i).write(item.clone());
                }
                array
            }
        }
    }
}

enum Order {
    Standard,
    Fortran,
}

impl Order {
    fn to_flag(&self) -> c_int {
        match self {
            Order::Standard => 0,
            Order::Fortran => 1,
        }
    }
}

trait ArrayExt {
    fn npy_strides(&self) -> NpyStrides;
    fn order(&self) -> Option<Order>;
}

impl<A, S, D> ArrayExt for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn npy_strides(&self) -> NpyStrides {
        NpyStrides::new(
            self.strides().iter().map(|&x| x as npyffi::npy_intp),
            mem::size_of::<A>(),
        )
    }

    fn order(&self) -> Option<Order> {
        if self.is_standard_layout() {
            Some(Order::Standard)
        } else if self.ndim() > 1 && self.raw_view().reversed_axes().is_standard_layout() {
            Some(Order::Fortran)
        } else {
            None
        }
    }
}

/// Numpy strides with short array optimization
enum NpyStrides {
    Short([npyffi::npy_intp; 8]),
    Long(Vec<npyffi::npy_intp>),
}

impl NpyStrides {
    fn as_ptr(&self) -> *const npy_intp {
        match self {
            NpyStrides::Short(inner) => inner.as_ptr(),
            NpyStrides::Long(inner) => inner.as_ptr(),
        }
    }
    fn from_dim<D: Dimension>(dim: &D, type_size: usize) -> Self {
        Self::new(
            dim.default_strides()
                .slice()
                .iter()
                .map(|&x| x as npyffi::npy_intp),
            type_size,
        )
    }
    fn new(strides: impl ExactSizeIterator<Item = npyffi::npy_intp>, type_size: usize) -> Self {
        let len = strides.len();
        let type_size = type_size as npyffi::npy_intp;
        if len <= 8 {
            let mut res = [0; 8];
            for (i, s) in strides.enumerate() {
                res[i] = s * type_size;
            }
            NpyStrides::Short(res)
        } else {
            NpyStrides::Long(strides.map(|n| n as npyffi::npy_intp * type_size).collect())
        }
    }
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
/// [IntoDimension](https://docs.rs/ndarray/latest/ndarray/dimension/conversion/trait.IntoDimension.html)
/// for what types you can use as `NpyIndex`.
///
/// But basically, you can use
/// - [tuple](https://doc.rust-lang.org/nightly/std/primitive.tuple.html)
/// - [array](https://doc.rust-lang.org/nightly/std/primitive.array.html)
/// - [slice](https://doc.rust-lang.org/nightly/std/primitive.slice.html)
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
        if indices.iter().zip(dims).any(|(i, d)| i >= d) {
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

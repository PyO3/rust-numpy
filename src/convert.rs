//! Defines conversion traits between rust types and numpy data types.

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix1, OwnedRepr};
use pyo3::Python;

use std::{mem, os::raw::c_int};

use crate::{
    npyffi::{self, npy_intp},
    Element, PyArray,
};

/// Conversion trait from some rust types to `PyArray`.
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
        let dims = [self.len()];
        let strides = [mem::size_of::<T>() as npy_intp];
        let data_ptr = self.as_ptr();
        unsafe { PyArray::from_raw_parts(py, dims, strides.as_ptr(), data_ptr, self) }
    }
}

impl<T: Element> IntoPyArray for Vec<T> {
    type Item = T;
    type Dim = Ix1;
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim> {
        let dims = [self.len()];
        let strides = [mem::size_of::<T>() as npy_intp];
        let data_ptr = self.as_ptr();
        unsafe { PyArray::from_raw_parts(py, dims, strides.as_ptr(), data_ptr, self) }
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
        PyArray::from_owned_array(py, self)
    }
}

/// Conversion trait from rust types to `PyArray`.
///
/// This trait takes `&self`, which means **it allocates in Python heap and then copies
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
        match self.order() {
            Some(order) if A::IS_COPY => {
                // if the array is contiguous, copy it by `copy_ptr`.
                let strides = self.npy_strides();
                unsafe {
                    let array =
                        PyArray::new_(py, self.raw_dim(), strides.as_ptr(), order.to_flag());
                    array.copy_ptr(self.as_ptr(), len);
                    array
                }
            }
            _ => {
                // if the array is not contiguous, copy all elements by `ArrayBase::iter`.
                let dim = self.raw_dim();
                let strides = NpyStrides::new::<_, A>(
                    dim.default_strides()
                        .slice()
                        .iter()
                        .map(|&x| x as npyffi::npy_intp),
                );
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
}

pub(crate) enum Order {
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

pub(crate) trait ArrayExt {
    fn npy_strides(&self) -> NpyStrides;
    fn order(&self) -> Option<Order>;
}

impl<A, S, D> ArrayExt for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn npy_strides(&self) -> NpyStrides {
        NpyStrides::new::<_, A>(self.strides().iter().map(|&x| x as npyffi::npy_intp))
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

/// An array of strides sufficiently large for [any NumPy array][NPY_MAXDIMS]
///
/// [NPY_MAXDIMS]: https://github.com/numpy/numpy/blob/4c60b3263ac50e5e72f6a909e156314fc3c9cba0/numpy/core/include/numpy/ndarraytypes.h#L40
pub(crate) struct NpyStrides([npyffi::npy_intp; 32]);

impl NpyStrides {
    pub(crate) fn as_ptr(&self) -> *const npy_intp {
        self.0.as_ptr()
    }

    fn new<S, A>(strides: S) -> Self
    where
        S: Iterator<Item = npyffi::npy_intp>,
    {
        let type_size = mem::size_of::<A>() as npyffi::npy_intp;
        let mut res = [0; 32];
        for (i, s) in strides.enumerate() {
            *res.get_mut(i)
                .expect("Only dimensionalities of up to 32 are supported") = s * type_size;
        }
        Self(res)
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

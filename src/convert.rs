//! Defines conversion traits between Rust types and NumPy data types.
#![deny(missing_docs)]

use std::{mem, os::raw::c_int, ptr};

use ndarray::{ArrayBase, Data, Dimension, IntoDimension, Ix1, OwnedRepr};
use pyo3::Python;

use crate::array::PyArray;
use crate::dtype::Element;
use crate::npyffi::{self, npy_intp};
use crate::sealed::Sealed;

/// Conversion trait from owning Rust types into [`PyArray`].
///
/// This trait takes ownership of `self`, which means it holds a pointer into the Rust heap.
///
/// In addition, some destructive methods like `resize` cannot be used with NumPy arrays constructed using this trait.
///
/// # Example
///
/// ```
/// use numpy::{PyArray, IntoPyArray};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].into_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
///
///     // Array cannot be resized when its data is owned by Rust.
///     unsafe {
///         assert!(py_array.resize(100).is_err());
///     }
/// });
/// ```
pub trait IntoPyArray {
    /// The element type of resulting array.
    type Item: Element;
    /// The dimension type of the resulting array.
    type Dim: Dimension;

    /// Consumes `self` and moves its data into a NumPy array.
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
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

/// Conversion trait from borrowing Rust types to [`PyArray`].
///
/// This trait takes `&self` by reference, which means it allocates in Python heap and then copies the elements there.
///
/// # Examples
///
/// ```
/// use numpy::{PyArray, ToPyArray};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let py_array = vec![1, 2, 3].to_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);
/// });
/// ```
///
/// Due to copying the elments, this method converts non-contiguous arrays to C-order contiguous arrays.
///
/// ```
/// use numpy::{PyArray, ToPyArray};
/// use ndarray::{arr3, s};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let array = arr3(&[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
///     let py_array = array.slice(s![.., 0..1, ..]).to_pyarray(py);
///
///     assert_eq!(py_array.readonly().as_array(), arr3(&[[[1, 2, 3]], [[7, 8, 9]]]));
///     assert!(py_array.is_c_contiguous());
/// });
/// ```
pub trait ToPyArray {
    /// The element type of resulting array.
    type Item: Element;
    /// The dimension type of the resulting array.
    type Dim: Dimension;

    /// Copies the content pointed to by `&self` into a newly allocated NumPy array.
    fn to_pyarray<'py>(&self, py: Python<'py>) -> &'py PyArray<Self::Item, Self::Dim>;
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
                // if the array is contiguous, copy it by `copy_nonoverlapping`.
                let strides = self.npy_strides();
                unsafe {
                    let array =
                        PyArray::new_(py, self.raw_dim(), strides.as_ptr(), order.to_flag());
                    ptr::copy_nonoverlapping(self.as_ptr(), array.data(), len);
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
                    let mut data_ptr = array.data();
                    for item in self.iter() {
                        data_ptr.write(item.clone());
                        data_ptr = data_ptr.add(1);
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

/// Utility trait to specify the dimensions of an array.
pub trait ToNpyDims: Dimension + Sealed {
    #[doc(hidden)]
    fn ndim_cint(&self) -> c_int {
        self.ndim() as c_int
    }
    #[doc(hidden)]
    fn as_dims_ptr(&self) -> *mut npyffi::npy_intp {
        self.slice().as_ptr() as *mut npyffi::npy_intp
    }
    #[doc(hidden)]
    fn to_npy_dims(&self) -> npyffi::PyArray_Dims {
        npyffi::PyArray_Dims {
            ptr: self.as_dims_ptr(),
            len: self.ndim_cint(),
        }
    }
}

impl<D> ToNpyDims for D where D: Dimension {}

/// Trait implemented by types that can be used to index an array.
///
/// This is equivalent to [`ndarray::NdIndex`] but accounts for
/// NumPy strides being in units of bytes instead of elements.
///
/// All types which implement [`IntoDimension`] implement this trait as well.
/// This includes at least
/// - [tuple](https://doc.rust-lang.org/stable/std/primitive.tuple.html)
/// - [array](https://doc.rust-lang.org/stable/std/primitive.array.html)
/// - [slice](https://doc.rust-lang.org/stable/std/primitive.slice.html)
pub trait NpyIndex: IntoDimension + Sealed {
    #[doc(hidden)]
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize>;
    #[doc(hidden)]
    fn get_unchecked<T>(self, strides: &[isize]) -> isize;
}

impl<D: IntoDimension> Sealed for D {}

impl<D: IntoDimension> NpyIndex for D {
    fn get_checked<T>(self, dims: &[usize], strides: &[isize]) -> Option<isize> {
        let indices = self.into_dimension();
        let indices = indices.slice();

        if indices.len() != dims.len() {
            return None;
        }
        if indices.iter().zip(dims).any(|(i, d)| i >= d) {
            return None;
        }

        Some(get_unchecked_impl::<T>(indices, strides))
    }

    fn get_unchecked<T>(self, strides: &[isize]) -> isize {
        let indices = self.into_dimension();
        let indices = indices.slice();
        get_unchecked_impl::<T>(indices, strides)
    }
}

fn get_unchecked_impl<T>(indices: &[usize], strides: &[isize]) -> isize {
    let size = mem::size_of::<T>() as isize;

    indices
        .iter()
        .zip(strides)
        .map(|(&i, stride)| stride * i as isize / size)
        .sum()
}

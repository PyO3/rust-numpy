//! Wrapper of [Array Iterator API](https://numpy.org/doc/stable/reference/c-api/iterator.html).
//!
//! This module exposes two iterators:
//! [NpySingleIter](./struct.NpySingleIter.html) and
//! [NpyMultiIter](./struct.NpyMultiIter.html).
use crate::npyffi::{
    array::PY_ARRAY_API,
    npy_intp, npy_uint32,
    objects::{NpyIter, PyArrayObject},
    types::{NPY_CASTING, NPY_ORDER},
    NPY_ARRAY_WRITEABLE, NPY_ITER_BUFFERED, NPY_ITER_COMMON_DTYPE, NPY_ITER_COPY_IF_OVERLAP,
    NPY_ITER_DELAY_BUFALLOC, NPY_ITER_DONT_NEGATE_STRIDES, NPY_ITER_GROWINNER, NPY_ITER_RANGED,
    NPY_ITER_READONLY, NPY_ITER_READWRITE, NPY_ITER_REDUCE_OK, NPY_ITER_REFS_OK,
    NPY_ITER_ZEROSIZE_OK,
};
use crate::{Element, PyArray, PyArrayDyn, PyReadonlyArray};
use pyo3::{prelude::*, PyNativeType};

use std::marker::PhantomData;
use std::os::raw::*;
use std::ptr;

/// Flags for constructing an iterator.
/// For the meanings of each flag, readers can refer to [the numpy document][doc].
///
/// Note that this enum doesn't provide all flags in the numpy C-API.
/// If you have any inconvenience about that, please file an [issue].
///
/// [doc]: https://numpy.org/doc/stable/reference/c-api/iterator.html#c.NpyIter_MultiNew
/// [issue]: https://github.com/PyO3/rust-numpy/issues
// Here's a list of unsupported flags:
// CIndex,
// FIndex,
// MultiIndex,
// ExternalLoop,
// ReadWrite,
// ReadOnly,
// WriteOnly,
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NpyIterFlag {
    CommonDtype,
    RefsOk,
    ZerosizeOk,
    ReduceOk,
    Ranged,
    Buffered,
    GrowInner,
    DelayBufAlloc,
    DontNegateStrides,
    CopyIfOverlap,
}

impl NpyIterFlag {
    fn to_c_enum(&self) -> npy_uint32 {
        use NpyIterFlag::*;
        match self {
            CommonDtype => NPY_ITER_COMMON_DTYPE,
            RefsOk => NPY_ITER_REFS_OK,
            ZerosizeOk => NPY_ITER_ZEROSIZE_OK,
            ReduceOk => NPY_ITER_REDUCE_OK,
            Ranged => NPY_ITER_RANGED,
            Buffered => NPY_ITER_BUFFERED,
            GrowInner => NPY_ITER_GROWINNER,
            DelayBufAlloc => NPY_ITER_DELAY_BUFALLOC,
            DontNegateStrides => NPY_ITER_DONT_NEGATE_STRIDES,
            CopyIfOverlap => NPY_ITER_COPY_IF_OVERLAP,
        }
    }
}

/// Defines IterMode and MultiIterMode.
mod itermode {
    use super::*;

    pub struct PrivateGuard;
    macro_rules! private_decl {
        () => {
            fn __private__() -> PrivateGuard;
        };
    }
    macro_rules! private_impl {
        () => {
            fn __private__() -> PrivateGuard {
                PrivateGuard
            }
        };
    }

    /// A combinator type that represents the mode of an iterator
    /// (E.g., Readonly, ReadWrite, Readonly + ReadWrite).
    pub trait MultiIterMode {
        private_decl!();
        type Pre: MultiIterMode;
        const FLAG: npy_uint32 = 0;
        fn flags() -> Vec<npy_uint32> {
            if Self::FLAG == 0 {
                vec![]
            } else {
                let mut res = Self::Pre::flags();
                res.push(Self::FLAG);
                res
            }
        }
    }

    impl MultiIterMode for () {
        private_impl!();
        type Pre = ();
    }

    /// Represents the iterator mode where the last array is readonly.
    pub struct RO<S: MultiIterMode>(PhantomData<S>);

    /// Represents the iterator mode where the last array is readwrite.
    pub struct RW<S: MultiIterMode>(PhantomData<S>);

    impl<S: MultiIterMode> MultiIterMode for RO<S> {
        private_impl!();
        type Pre = S;
        const FLAG: npy_uint32 = NPY_ITER_READONLY;
    }

    impl<S: MultiIterMode> MultiIterMode for RW<S> {
        private_impl!();
        type Pre = S;
        const FLAG: npy_uint32 = NPY_ITER_READWRITE;
    }

    /// Represents the iterator mode where at least two arrays are iterated.
    pub trait MultiIterModeWithManyArrays: MultiIterMode {}
    impl MultiIterModeWithManyArrays for RO<RO<()>> {}
    impl MultiIterModeWithManyArrays for RO<RW<()>> {}
    impl MultiIterModeWithManyArrays for RW<RO<()>> {}
    impl MultiIterModeWithManyArrays for RW<RW<()>> {}

    impl<S: MultiIterModeWithManyArrays> MultiIterModeWithManyArrays for RO<S> {}
    impl<S: MultiIterModeWithManyArrays> MultiIterModeWithManyArrays for RW<S> {}

    /// Iterator mode for single iterator
    pub trait IterMode: MultiIterMode {}
    /// Implies Readonly iterator.
    pub type Readonly = RO<()>;
    /// Implies Readwrite iterator.
    pub type ReadWrite = RW<()>;

    impl IterMode for RO<()> {}
    impl IterMode for RW<()> {}
}

pub use itermode::{
    IterMode, MultiIterMode, MultiIterModeWithManyArrays, ReadWrite, Readonly, RO, RW,
};

/// Builder of [NpySingleIter](./struct.NpySingleIter.html).
pub struct NpySingleIterBuilder<'py, T, I: IterMode> {
    flags: npy_uint32,
    array: &'py PyArrayDyn<T>,
    mode: PhantomData<I>,
    was_writable: bool,
}

impl<'py, T: Element> NpySingleIterBuilder<'py, T, Readonly> {
    /// Makes a new builder for a readonly iterator.
    pub fn readonly<D: ndarray::Dimension>(array: PyReadonlyArray<'py, T, D>) -> Self {
        let (array, was_writable) = array.destruct();
        Self {
            flags: NPY_ITER_READONLY,
            array: array.to_dyn(),
            mode: PhantomData,
            was_writable,
        }
    }
}

impl<'py, T: Element> NpySingleIterBuilder<'py, T, ReadWrite> {
    /// Makes a new builder for a writable iterator.
    pub fn readwrite<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> Self {
        Self {
            flags: NPY_ITER_READWRITE,
            array: array.to_dyn(),
            mode: PhantomData,
            was_writable: false,
        }
    }
}

impl<'py, T: Element, I: IterMode> NpySingleIterBuilder<'py, T, I> {
    /// Sets a flag to this builder, returning `self`.
    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }

    /// Creates an iterator from this builder.
    pub fn build(self) -> PyResult<NpySingleIter<'py, T, I>> {
        let array_ptr = self.array.as_array_ptr();
        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_New(
                array_ptr,
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                ptr::null_mut(),
            )
        };
        let py = self.array.py();
        let readonly_array_ptr = if self.was_writable {
            Some(array_ptr)
        } else {
            None
        };
        NpySingleIter::new(iter_ptr, readonly_array_ptr, py)
    }
}

/// An iterator over a single array, construced by
/// [NpySingleIterBuilder](./struct.NpySingleIterBuilder.html).
/// This iterator iterates all elements in the array as `&mut T` (in case `readwrite` is used)
/// or `&T` (in case `readonly` is used).
///
/// # Example
///
/// You can use
/// [`NpySingleIterBuilder::readwrite`](./struct.NpySingleIterBuilder.html#method.readwrite)
/// to get a mutable iterator.
///
/// ```
/// use numpy::NpySingleIterBuilder;
/// pyo3::Python::with_gil(|py| {
///     let array = numpy::PyArray::arange(py, 0, 10, 1);
///     let iter = NpySingleIterBuilder::readwrite(array).build().unwrap();
///     for (i, elem) in iter.enumerate() {
///         assert_eq!(*elem, i as i64);
///         *elem = *elem * 2;  // elements are mutable
///     }
/// });
/// ```
/// Or, as a shorthand, `PyArray::iter` can be also used.
/// ```
/// # use numpy::NpySingleIterBuilder;
/// # pyo3::Python::with_gil(|py| {
/// #   let array = numpy::PyArray::arange(py, 0, 10, 1);
/// #   let iter = NpySingleIterBuilder::readwrite(array).build().unwrap();
///     for (i, elem) in array.iter().unwrap().enumerate() {
///         assert_eq!(*elem, i as i64);
///         *elem = *elem * 2;  // elements are mutable
///     }
/// });
/// ```
/// On the other hand, immutable iterator requires [readonly array](../struct.PyReadonlyArray.html).
/// ```
/// use numpy::NpySingleIterBuilder;
/// pyo3::Python::with_gil(|py| {
///     let array = numpy::PyArray::arange(py, 0, 1, 10);
///     let iter = NpySingleIterBuilder::readonly(array.readonly()).build().unwrap();
///     for (i, elem) in iter.enumerate() {
///         assert_eq!(*elem, i as i64);
///     }
/// });
/// ```
pub struct NpySingleIter<'py, T, I> {
    iterator: ptr::NonNull<NpyIter>,
    iternext: unsafe extern "C" fn(*mut NpyIter) -> c_int,
    empty: bool,
    iter_size: npy_intp,
    dataptr: *mut *mut c_char,
    return_type: PhantomData<T>,
    mode: PhantomData<I>,
    readonly_array_ptr: Option<*mut PyArrayObject>,
    _py: Python<'py>,
}

impl<'py, T, I> NpySingleIter<'py, T, I> {
    fn new(
        iterator: *mut NpyIter,
        readonly_array_ptr: Option<*mut PyArrayObject>,
        py: Python<'py>,
    ) -> PyResult<Self> {
        let mut iterator = match ptr::NonNull::new(iterator) {
            Some(iter) => iter,
            None => {
                return Err(PyErr::fetch(py));
            }
        };

        let iternext =
            match unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut()) } {
                Some(ptr) => ptr,
                None => {
                    return Err(PyErr::fetch(py));
                }
            };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
            return Err(PyErr::fetch(py));
        }

        let iter_size = unsafe { PY_ARRAY_API.NpyIter_GetIterSize(iterator.as_mut()) };

        Ok(Self {
            iterator,
            iternext,
            iter_size,
            empty: iter_size == 0,
            dataptr,
            return_type: PhantomData,
            mode: PhantomData,
            readonly_array_ptr,
            _py: py,
        })
    }

    fn iternext(&mut self) -> Option<*mut T> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let ret = unsafe { *self.dataptr as *mut T };
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            Some(ret)
        }
    }
}

impl<'py, T, I> Drop for NpySingleIter<'py, T, I> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        if let Some(ptr) = self.readonly_array_ptr {
            unsafe {
                (*ptr).flags |= NPY_ARRAY_WRITEABLE;
            }
        }
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpySingleIter<'py, T, Readonly> {
    type Item = &'py T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iternext().map(|ptr| unsafe { &*ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.iter_size as usize, Some(self.iter_size as usize))
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpySingleIter<'py, T, ReadWrite> {
    type Item = &'py mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iternext().map(|ptr| unsafe { &mut *ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.iter_size as usize, Some(self.iter_size as usize))
    }
}

/// Builder for [NpyMultiIter](./struct.NpyMultiIter.html).
pub struct NpyMultiIterBuilder<'py, T, S: MultiIterMode> {
    flags: npy_uint32,
    arrays: Vec<&'py PyArrayDyn<T>>,
    structure: PhantomData<S>,
    was_writables: Vec<bool>,
}

impl<'py, T: Element> Default for NpyMultiIterBuilder<'py, T, ()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'py, T: Element> NpyMultiIterBuilder<'py, T, ()> {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            flags: 0,
            arrays: Vec::new(),
            structure: PhantomData,
            was_writables: Vec::new(),
        }
    }

    /// Set a flag to this builder, returning `self`.
    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }
}

impl<'py, T: Element, S: MultiIterMode> NpyMultiIterBuilder<'py, T, S> {
    /// Add a readonly array to the resulting iterator.
    pub fn add_readonly<D: ndarray::Dimension>(
        mut self,
        array: PyReadonlyArray<'py, T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RO<S>> {
        let (array, was_writable) = array.destruct();
        self.arrays.push(array.to_dyn());
        self.was_writables.push(was_writable);
        NpyMultiIterBuilder {
            flags: self.flags,
            arrays: self.arrays,
            was_writables: self.was_writables,
            structure: PhantomData,
        }
    }

    /// Adds a writable array to the resulting iterator.
    pub fn add_readwrite<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RW<S>> {
        self.arrays.push(array.to_dyn());
        self.was_writables.push(false);
        NpyMultiIterBuilder {
            flags: self.flags,
            arrays: self.arrays,
            was_writables: self.was_writables,
            structure: PhantomData,
        }
    }
}

impl<'py, T: Element, S: MultiIterModeWithManyArrays> NpyMultiIterBuilder<'py, T, S> {
    /// Creates an iterator from this builder.
    pub fn build(self) -> PyResult<NpyMultiIter<'py, T, S>> {
        let Self {
            flags,
            arrays,
            was_writables,
            ..
        } = self;
        debug_assert!(arrays.len() <= std::i32::MAX as usize);
        debug_assert!(2 <= arrays.len());

        let mut opflags = S::flags();
        let py = arrays[0].py();
        let mut arrays = arrays
            .iter()
            .map(|x| x.as_array_ptr())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_MultiNew(
                arrays.len() as i32,
                arrays.as_mut_ptr(),
                flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                opflags.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        NpyMultiIter::new(iter_ptr, arrays, was_writables, py)
    }
}

/// An iterator over multiple arrays, construced by
/// [NpyMultiIterBuilder](./struct.NpyMultiIterBuilder.html).
/// You can add
/// [`NpyMultiIterBuilder::add_readwrite`](./struct.NpyMultiIterBuilder.html#method.add_readwrite)
/// for adding a mutable component to the iterator, and
/// [`NpyMultiIterBuilder::add_readonly`](./struct.NpyMultiIterBuilder.html#method.add_readonly)
/// for adding a immutable one.
///
/// # Example
///
/// ```
/// use numpy::NpyMultiIterBuilder;
/// pyo3::Python::with_gil(|py| {
///     let array1 = numpy::PyArray::arange(py, 0, 10, 1);
///     let array2 = numpy::PyArray::arange(py, 10, 20, 1);
///     let array3 = numpy::PyArray::arange(py, 10, 30, 2);
///     let iter = NpyMultiIterBuilder::new()
///         .add_readonly(array1.readonly())
///         .add_readwrite(array2)
///         .add_readonly(array3.readonly())
///         .build()
///          .unwrap();
///     for (i, j, k) in iter {
///         assert_eq!(*i + *j, *k);
///         *j += *i + *k;  // The third element is only mutable.
///     }
/// });
/// ```
pub struct NpyMultiIter<'py, T, S: MultiIterModeWithManyArrays> {
    iterator: ptr::NonNull<NpyIter>,
    iternext: unsafe extern "C" fn(*mut NpyIter) -> c_int,
    empty: bool,
    iter_size: npy_intp,
    dataptr: *mut *mut c_char,
    marker: PhantomData<(T, S)>,
    arrays: Box<[*mut PyArrayObject]>,
    was_writables: Vec<bool>,
    _py: Python<'py>,
}

impl<'py, T, S: MultiIterModeWithManyArrays> NpyMultiIter<'py, T, S> {
    fn new(
        iterator: *mut NpyIter,
        arrays: Box<[*mut PyArrayObject]>,
        was_writables: Vec<bool>,
        py: Python<'py>,
    ) -> PyResult<Self> {
        let mut iterator = match ptr::NonNull::new(iterator) {
            Some(ptr) => ptr,
            None => {
                return Err(PyErr::fetch(py));
            }
        };

        let iternext =
            match unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut()) } {
                Some(ptr) => ptr,
                None => {
                    return Err(PyErr::fetch(py));
                }
            };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
            return Err(PyErr::fetch(py));
        }

        let iter_size = unsafe { PY_ARRAY_API.NpyIter_GetIterSize(iterator.as_mut()) };

        Ok(Self {
            iterator,
            iternext,
            iter_size,
            empty: iter_size == 0,
            dataptr,
            marker: PhantomData,
            arrays,
            was_writables,
            _py: py,
        })
    }
}

impl<'py, T, S: MultiIterModeWithManyArrays> Drop for NpyMultiIter<'py, T, S> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        for (array_ptr, &was_writable) in self.arrays.iter().zip(self.was_writables.iter()) {
            if was_writable {
                unsafe { (**array_ptr).flags |= NPY_ARRAY_WRITEABLE };
            }
        }
    }
}

macro_rules! impl_multi_iter {
    ($structure: ty, $($ty: ty)+, $($ptr: ident)+, $expand: ident, $deref: expr) => {
        impl<'py, T: 'py> std::iter::Iterator for NpyMultiIter<'py, T, $structure> {
            type Item = ($($ty,)+);
            fn next(&mut self) -> Option<Self::Item> {
                if self.empty {
                    None
                } else {
                    // Note: This pointer is correct and doesn't need to be updated,
                    // note that we're derefencing a **char into a *char casting to a *T
                    // and then transforming that into a reference, the value that dataptr
                    // points to is being updated by iternext to point to the next value.
                    let ($($ptr,)+) = unsafe { $expand::<T>(self.dataptr) };
                    let retval = Some(unsafe { $deref });
                    self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
                    retval
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.iter_size as usize, Some(self.iter_size as usize))
            }
        }
    };
}

// Helper functions for conversion
#[inline(always)]
unsafe fn expand2<T>(dataptr: *mut *mut c_char) -> (*mut T, *mut T) {
    (*dataptr as *mut T, *dataptr.offset(1) as *mut T)
}

#[inline(always)]
unsafe fn expand3<T>(dataptr: *mut *mut c_char) -> (*mut T, *mut T, *mut T) {
    (
        *dataptr as *mut T,
        *dataptr.offset(1) as *mut T,
        *dataptr.offset(2) as *mut T,
    )
}

impl_multi_iter!(RO<RO<()>>, &'py T &'py T, a b, expand2, (&*a, &*b));
impl_multi_iter!(RO<RW<()>>, &'py mut T &'py T, a b, expand2, (&mut *a, &*b));
impl_multi_iter!(RW<RO<()>>, &'py T &'py mut T, a b, expand2, (&*a, &mut *b));
impl_multi_iter!(RW<RW<()>>, &'py mut T &'py mut T, a b, expand2, (&mut *a, &mut *b));
impl_multi_iter!(RO<RO<RO<()>>>, &'py T &'py T &'py T, a b c, expand3, (&*a, &*b, &*c));
impl_multi_iter!(RO<RO<RW<()>>>, &'py mut T &'py T &'py T, a b c, expand3, (&mut *a, &*b, &*c));
impl_multi_iter!(RO<RW<RO<()>>>, &'py T &'py mut T &'py T, a b c, expand3, (&*a, &mut *b, &*c));
impl_multi_iter!(RW<RO<RO<()>>>, &'py T &'py T &'py mut T, a b c, expand3, (&*a, &*b, &mut *c));
impl_multi_iter!(RO<RW<RW<()>>>, &'py mut T &'py mut T &'py T, a b c, expand3, (&mut *a, &mut *b, &*c));
impl_multi_iter!(RW<RO<RW<()>>>, &'py mut T &'py T &'py mut T, a b c, expand3, (&mut *a, &*b, &mut *c));
impl_multi_iter!(RW<RW<RO<()>>>, &'py T &'py mut T &'py mut T, a b c, expand3, (&*a, &mut *b, &mut *c));
impl_multi_iter!(RW<RW<RW<()>>>, &'py mut T &'py mut T &'py mut T, a b c, expand3, (&mut *a, &mut *b, &mut *c));

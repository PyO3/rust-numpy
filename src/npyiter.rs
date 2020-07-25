use crate::array::{PyArray, PyArrayDyn};
use crate::npyffi::{
    array::PY_ARRAY_API,
    types::{NPY_CASTING, NPY_ORDER},
    *,
};
use crate::types::Element;
use crate::error::NpyIterInstantiationError;
use pyo3::{prelude::*, PyNativeType};

use std::marker::PhantomData;
use std::os::raw::*;
use std::ptr;

/// Flags for `NpySingleIter` and `NpyMultiIter`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NpyIterFlag {
    /* CIndex,
    FIndex,
    MultiIndex, */
    // ExternalLoop, // This flag greatly modifies the behaviour of accessing the data
    // so we don't support it.
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
    /* ReadWrite,
    ReadOnly,
    WriteOnly, */
}

impl NpyIterFlag {
    fn to_c_enum(&self) -> npy_uint32 {
        use NpyIterFlag::*;
        match self {
            /* CIndex => NPY_ITER_C_INDEX,
            FIndex => NPY_ITER_C_INDEX,
            MultiIndex => NPY_ITER_MULTI_INDEX, */
            /* ExternalLoop => NPY_ITER_EXTERNAL_LOOP, */
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
            /* ReadWrite => NPY_ITER_READWRITE,
            ReadOnly => NPY_ITER_READONLY,
            WriteOnly => NPY_ITER_WRITEONLY, */
        }
    }
}

pub struct NpySingleIterBuilder<'py, T> {
    flags: npy_uint32,
    array: &'py PyArrayDyn<T>,
}

impl<'py, T: Element> NpySingleIterBuilder<'py, T> {
    pub fn readwrite<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> Self {
        Self {
            flags: NPY_ITER_READWRITE,
            array: array.to_dyn(),
        }
    }

    pub fn readonly<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> Self {
        Self {
            flags: NPY_ITER_READONLY,
            array: array.to_dyn(),
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }

    pub fn build(self) -> PyResult<NpySingleIter<'py, T>> {
        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_New(
                self.array.as_array_ptr(),
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                ptr::null_mut(),
            )
        };
        let py = self.array.py();
        NpySingleIter::new(iter_ptr, py)
    }
}

pub struct NpySingleIter<'py, T> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    dataptr: *mut *mut c_char,
    return_type: PhantomData<T>,
    _py: Python<'py>,
}

impl<'py, T> NpySingleIter<'py, T> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> PyResult<NpySingleIter<'py, T>> {
        let mut iterator = match ptr::NonNull::new(iterator) {
            Some(iter) => iter,
            None => {
                return Err(NpyIterInstantiationError.into());
            }
        };

        // TODO replace the null second arg with something correct.
        let iternext = match unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut()) } {
            Some(ptr) => ptr,
            None => {
                return Err(PyErr::fetch(py));
            }
        };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
            return Err(NpyIterInstantiationError.into());
        }

        Ok(NpySingleIter {
            iterator,
            iternext,
            empty: false, // TODO: Handle empty iterators
            dataptr,
            return_type: PhantomData,
            _py: py,
        })
    }
}

impl<'py, T> Drop for NpySingleIter<'py, T> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpySingleIter<'py, T> {
    type Item = &'py T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe { &*(*self.dataptr as *mut T) });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

mod private {
    pub struct PrivateGuard;
}
macro_rules! private_decl {
    () => {
        fn __private__() -> private::PrivateGuard;
    };
}
macro_rules! private_impl {
    () => {
        fn __private__() -> private::PrivateGuard {
            private::PrivateGuard
        }
    };
}

/// A combinator type that represents an terator mode (e.g., ReadOnly + ReadWrite + ReadOnly).
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

impl<S: MultiIterMode> MultiIterMode for RO<S> {
    private_impl!();
    type Pre = S;
    const FLAG: npy_uint32 = NPY_ITER_READONLY;
}

/// Represents the iterator mode where the last array is readwrite.
pub struct RW<S: MultiIterMode>(PhantomData<S>);

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

/// A builder struct for creating multi iterator.
pub struct NpyMultiIterBuilder<'py, T, S: MultiIterMode> {
    flags: npy_uint32,
    arrays: Vec<&'py PyArrayDyn<T>>,
    structure: PhantomData<S>,
}

impl<'py, T: Element> NpyMultiIterBuilder<'py, T, ()> {
    pub fn new() -> Self {
        Self {
            flags: 0,
            arrays: Vec::new(),
            structure: PhantomData,
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }
}

impl<'py, T: Element, S: MultiIterMode> NpyMultiIterBuilder<'py, T, S> {
    pub fn add_readonly_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RO<S>> {
        self.arrays.push(array.to_dyn());

        NpyMultiIterBuilder {
            flags: self.flags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }

    pub fn add_readwrite_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RW<S>> {
        self.arrays.push(array.to_dyn());

        NpyMultiIterBuilder {
            flags: self.flags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }
}

impl<'py, T: Element, S: MultiIterModeWithManyArrays> NpyMultiIterBuilder<'py, T, S> {
    pub fn build(mut self) -> PyResult<NpyMultiIter<'py, T, S>> {
        assert!(self.arrays.len() <= i32::MAX as usize);
        assert!(2 <= self.arrays.len());

        let mut opflags = S::flags();

        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_MultiNew(
                self.arrays.len() as i32,
                self.arrays
                    .iter_mut()
                    .map(|x| x.as_array_ptr())
                    .collect::<Vec<_>>()
                    .as_mut_ptr(),
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                opflags.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        let py = self.arrays[0].py();
        NpyMultiIter::new(iter_ptr, py).ok_or_else(|| PyErr::fetch(py))
    }
}

/// Multi iterator
pub struct NpyMultiIter<'py, T, S: MultiIterModeWithManyArrays> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    iter_size: npy_intp,
    dataptr: *mut *mut c_char,
    marker: PhantomData<(T, S)>,
    _py: Python<'py>,
}

impl<'py, T, S: MultiIterModeWithManyArrays> NpyMultiIter<'py, T, S> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> Option<Self> {
        let mut iterator = ptr::NonNull::new(iterator)?;

        // TODO replace the null second arg with something correct.
        let iternext =
            unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut())? };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
        }

        let iter_size = unsafe { PY_ARRAY_API.NpyIter_GetIterSize(iterator.as_mut()) };

        Some(Self {
            iterator,
            iternext,
            iter_size,
            empty: iter_size == 0, // TODO: Handle empty iterators
            dataptr,
            marker: PhantomData,
            _py: py,
        })
    }
}

impl<'py, T, S: MultiIterModeWithManyArrays> Drop for NpyMultiIter<'py, T, S> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
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

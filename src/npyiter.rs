use crate::array::{PyArray, PyArrayDyn};
use crate::npyffi::{
    array::PY_ARRAY_API,
    types::{NPY_CASTING, NPY_ORDER},
    *,
};
use crate::types::TypeNum;
use pyo3::{prelude::*, PyNativeType};

use std::marker::PhantomData;
use std::os::raw::*;
use std::ptr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NpyIterFlag {
    CIndex,
    FIndex,
    MultiIndex,
    ExternalLoop,
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
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

impl NpyIterFlag {
    fn to_c_enum(&self) -> npy_uint32 {
        use NpyIterFlag::*;
        match self {
            CIndex => NPY_ITER_C_INDEX,
            FIndex => NPY_ITER_C_INDEX,
            MultiIndex => NPY_ITER_MULTI_INDEX,
            ExternalLoop => NPY_ITER_EXTERNAL_LOOP,
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
            ReadWrite => NPY_ITER_READWRITE,
            ReadOnly => NPY_ITER_READONLY,
            WriteOnly => NPY_ITER_WRITEONLY,
        }
    }
}

pub struct NpyIterBuilder<'py, T> {
    flags: npy_uint32,
    array: &'py PyArrayDyn<T>,
}

impl<'py, T: TypeNum> NpyIterBuilder<'py, T> {
    pub fn new<D: ndarray::Dimension>(array: &'py PyArray<T, D>) -> NpyIterBuilder<'py, T> {
        NpyIterBuilder {
            flags: 0,
            array: array.into_dyn(),
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        if flag == NpyIterFlag::ExternalLoop {
            // TODO: I don't want to make set fallible, but also we don't want to
            // support ExternalLoop yet (maybe ever?).
            panic!("rust-numpy does not currently support ExternalLoop access");
        }
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }

    pub fn build(self) -> PyResult<NpyIterSingleArray<'py, T>> {
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
        NpyIterSingleArray::new(iter_ptr, py).ok_or_else(|| PyErr::fetch(py))
    }
}

pub struct NpyIterSingleArray<'py, T> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    dataptr: *mut *mut c_char,

    return_type: PhantomData<T>,
    _py: Python<'py>,
}

impl<'py, T> NpyIterSingleArray<'py, T> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> Option<NpyIterSingleArray<'py, T>> {
        let mut iterator = ptr::NonNull::new(iterator)?;

        // TODO replace the null second arg with something correct.
        let iternext =
            unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut())? };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
        }

        Some(NpyIterSingleArray {
            iterator,
            iternext,
            empty: false, // TODO: Handle empty iterators
            dataptr,
            return_type: PhantomData,
            _py: py,
        })
    }
}

impl<'py, T> Drop for NpyIterSingleArray<'py, T> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpyIterSingleArray<'py, T> {
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

pub trait MultiIterMode {}

impl MultiIterMode for () {}

pub struct RO<S> {
    structure: PhantomData<S>,
}

impl<S: MultiIterMode> MultiIterMode for RO<S> {}

pub struct RW<S> {
    structure: PhantomData<S>,
}

impl<S: MultiIterMode> MultiIterMode for RW<S> {}

pub trait MultiIterModeHasManyArrays: MultiIterMode {}
impl MultiIterModeHasManyArrays for RO<RO<()>> {}
impl MultiIterModeHasManyArrays for RO<RW<()>> {}
impl MultiIterModeHasManyArrays for RW<RO<()>> {}
impl MultiIterModeHasManyArrays for RW<RW<()>> {}

impl<S: MultiIterModeHasManyArrays> MultiIterModeHasManyArrays for RO<S> {}
impl<S: MultiIterModeHasManyArrays> MultiIterModeHasManyArrays for RW<S> {}

pub struct NpyMultiIterBuilder<'py, T, S: MultiIterMode> {
    flags: npy_uint32,
    opflags: Vec<npy_uint32>,
    arrays: Vec<&'py PyArrayDyn<T>>,
    structure: PhantomData<S>,
}

impl<'py, T: TypeNum> NpyMultiIterBuilder<'py, T, ()> {
    pub fn new() -> Self {
        Self {
            flags: 0,
            opflags: Vec::new(),
            arrays: Vec::new(),
            structure: PhantomData,
        }
    }

    pub fn set(mut self, flag: NpyIterFlag) -> Self {
        if flag == NpyIterFlag::ExternalLoop {
            // TODO: I don't want to make set fallible, but also we don't want to
            // support ExternalLoop yet (maybe ever?).
            panic!("rust-numpy does not currently support ExternalLoop access");
        }
        self.flags |= flag.to_c_enum();
        self
    }

    pub fn unset(mut self, flag: NpyIterFlag) -> Self {
        self.flags &= !flag.to_c_enum();
        self
    }
}

impl<'py, T: TypeNum, S: MultiIterMode> NpyMultiIterBuilder<'py, T, S> {
    pub fn add_readonly_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RO<S>> {
        self.arrays.push(array.into_dyn());
        self.opflags.push(NPY_ITER_READONLY);

        NpyMultiIterBuilder {
            flags: self.flags,
            opflags: self.opflags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }

    pub fn add_readwrite_array<D: ndarray::Dimension>(
        mut self,
        array: &'py PyArray<T, D>,
    ) -> NpyMultiIterBuilder<'py, T, RW<S>> {
        self.arrays.push(array.into_dyn());
        self.opflags.push(NPY_ITER_READWRITE);

        NpyMultiIterBuilder {
            flags: self.flags,
            opflags: self.opflags,
            arrays: self.arrays,
            structure: PhantomData,
        }
    }
}

impl<'py, T: TypeNum, S: MultiIterModeHasManyArrays> NpyMultiIterBuilder<'py, T, S> {
    pub fn build(mut self) -> PyResult<NpyMultiIterArray<'py, T, S>> {
        assert!(self.arrays.len() == self.opflags.len());
        assert!(self.arrays.len() <= i32::MAX as usize);
        assert!(2 <= self.arrays.len());

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
                self.opflags.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        let py = self.arrays[0].py();
        NpyMultiIterArray::new(iter_ptr, py).ok_or_else(|| PyErr::fetch(py))
    }
}

pub struct NpyMultiIterArray<'py, T, S: MultiIterModeHasManyArrays> {
    iterator: ptr::NonNull<objects::NpyIter>,
    iternext: unsafe extern "C" fn(*mut objects::NpyIter) -> c_int,
    empty: bool,
    dataptr: *mut *mut c_char,

    return_type: PhantomData<T>,
    structure: PhantomData<S>,
    _py: Python<'py>,
}

impl<'py, T, S: MultiIterModeHasManyArrays> NpyMultiIterArray<'py, T, S> {
    fn new(iterator: *mut objects::NpyIter, py: Python<'py>) -> Option<Self> {
        let mut iterator = ptr::NonNull::new(iterator)?;

        // TODO replace the null second arg with something correct.
        let iternext =
            unsafe { PY_ARRAY_API.NpyIter_GetIterNext(iterator.as_mut(), ptr::null_mut())? };
        let dataptr = unsafe { PY_ARRAY_API.NpyIter_GetDataPtrArray(iterator.as_mut()) };

        if dataptr.is_null() {
            unsafe { PY_ARRAY_API.NpyIter_Deallocate(iterator.as_mut()) };
        }

        Some(Self {
            iterator,
            iternext,
            empty: false, // TODO: Handle empty iterators
            dataptr,
            return_type: PhantomData,
            structure: PhantomData,
            _py: py,
        })
    }
}

impl<'py, T, S: MultiIterModeHasManyArrays> Drop for NpyMultiIterArray<'py, T, S> {
    fn drop(&mut self) {
        let _success = unsafe { PY_ARRAY_API.NpyIter_Deallocate(self.iterator.as_mut()) };
        // TODO: Handle _success somehow?
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpyMultiIterArray<'py, T, RO<RO<()>>> {
    type Item = (&'py T, &'py T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe {
                (
                    &*(*self.dataptr as *mut T),
                    &*(*self.dataptr.offset(1) as *mut T),
                )
            });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpyMultiIterArray<'py, T, RO<RW<()>>> {
    type Item = (&'py mut T, &'py T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe {
                (
                    &mut *(*self.dataptr as *mut T),
                    &*(*self.dataptr.offset(1) as *mut T),
                )
            });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpyMultiIterArray<'py, T, RW<RO<()>>> {
    type Item = (&'py T, &'py mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe {
                (
                    &*(*self.dataptr as *mut T),
                    &mut *(*self.dataptr.offset(1) as *mut T),
                )
            });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

impl<'py, T: 'py> std::iter::Iterator for NpyMultiIterArray<'py, T, RW<RW<()>>> {
    type Item = (&'py mut T, &'py mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.empty {
            None
        } else {
            // Note: This pointer is correct and doesn't need to be updated,
            // note that we're derefencing a **char into a *char casting to a *T
            // and then transforming that into a reference, the value that dataptr
            // points to is being updated by iternext to point to the next value.
            let retval = Some(unsafe {
                (
                    &mut *(*self.dataptr as *mut T),
                    &mut *(*self.dataptr.offset(1) as *mut T),
                )
            });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

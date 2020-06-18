use crate::array::PyArray;
use crate::npyffi;
use crate::npyffi::array::PY_ARRAY_API;
use crate::npyffi::objects;
use crate::npyffi::types::{npy_uint32, NPY_CASTING, NPY_ORDER};
use pyo3::prelude::*;

use std::marker::PhantomData;
use std::os::raw::*;
use std::ptr;

pub enum NPyIterFlag {
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
}

/*

#define NPY_ITER_C_INDEX                    0x00000001
#define NPY_ITER_F_INDEX                    0x00000002
#define NPY_ITER_MULTI_INDEX                0x00000004
#define NPY_ITER_EXTERNAL_LOOP              0x00000008
#define NPY_ITER_COMMON_DTYPE               0x00000010
#define NPY_ITER_REFS_OK                    0x00000020
#define NPY_ITER_ZEROSIZE_OK                0x00000040
#define NPY_ITER_REDUCE_OK                  0x00000080
#define NPY_ITER_RANGED                     0x00000100
#define NPY_ITER_BUFFERED                   0x00000200
#define NPY_ITER_GROWINNER                  0x00000400
#define NPY_ITER_DELAY_BUFALLOC             0x00000800
#define NPY_ITER_DONT_NEGATE_STRIDES        0x00001000
#define NPY_ITER_COPY_IF_OVERLAP            0x00002000
#define NPY_ITER_READWRITE                  0x00010000
#define NPY_ITER_READONLY                   0x00020000
#define NPY_ITER_WRITEONLY                  0x00040000
#define NPY_ITER_NBO                        0x00080000
#define NPY_ITER_ALIGNED                    0x00100000
#define NPY_ITER_CONTIG                     0x00200000
#define NPY_ITER_COPY                       0x00400000
#define NPY_ITER_UPDATEIFCOPY               0x00800000
#define NPY_ITER_ALLOCATE                   0x01000000
#define NPY_ITER_NO_SUBTYPE                 0x02000000
#define NPY_ITER_VIRTUAL                    0x04000000
#define NPY_ITER_NO_BROADCAST               0x08000000
#define NPY_ITER_WRITEMASKED                0x10000000
#define NPY_ITER_ARRAYMASK                  0x20000000
#define NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE 0x40000000

#define NPY_ITER_GLOBAL_FLAGS               0x0000ffff
#define NPY_ITER_PER_OP_FLAGS               0xffff0000

*/

impl NPyIterFlag {
    fn to_c_enum(&self) -> npy_uint32 {
        use NPyIterFlag::*;
        match self {
            CIndex => 0x00000001,
            FIndex => 0x00000002,
            MultiIndex => 0x00000004,
            ExternalLoop => 0x00000008,
            CommonDtype => 0x00000010,
            RefsOk => 0x00000020,
            ZerosizeOk => 0x00000040,
            ReduceOk => 0x00000080,
            Ranged => 0x00000100,
            Buffered => 0x00000200,
            GrowInner => 0x00000400,
            DelayBufAlloc => 0x00000800,
            DontNegateStrides => 0x00001000,
            CopyIfOverlap => 0x00002000,
        }
    }
}

pub struct NpyIterBuilder<'py, T> {
    flags: npy_uint32,
    array: *mut npyffi::PyArrayObject,
    py: Python<'py>,
    return_type: PhantomData<T>,
}

impl<'py, T> NpyIterBuilder<'py, T> {
    pub fn new<D>(array: PyArray<T, D>, py: Python<'py>) -> NpyIterBuilder<'py, T> {
        NpyIterBuilder {
            array: array.as_array_ptr(),
            py,
            flags: 0,
            return_type: PhantomData,
        }
    }

    pub fn set_iter_flags(&mut self, flag: NPyIterFlag, value: bool) -> &mut Self {
        if value {
            self.flags |= flag.to_c_enum();
        } else {
            self.flags &= !flag.to_c_enum();
        }
        self
    }

    pub fn finish(self) -> Option<NpyIterSingleArray<'py, T>> {
        let iter_ptr = unsafe {
            PY_ARRAY_API.NpyIter_New(
                self.array,
                self.flags,
                NPY_ORDER::NPY_ANYORDER,
                NPY_CASTING::NPY_SAFE_CASTING,
                ptr::null_mut(),
            )
        };

        NpyIterSingleArray::new(iter_ptr, self.py)
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
            let retval = Some(unsafe { &*(*self.dataptr as *mut T) });
            self.empty = unsafe { (self.iternext)(self.iterator.as_mut()) } == 0;
            retval
        }
    }
}

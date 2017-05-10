#![allow(non_camel_case_types)]

use pyffi::*;
use std::os::raw::*;
use std::option::Option;

use super::types::*;
use super::array::*;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NpyIter([u8; 0]);

pub type NpyIter_IterNextFunc = Option<unsafe extern "C" fn(iter: *mut NpyIter) -> c_int>;

pub type NpyIter_GetMultiIndexFunc = Option<unsafe extern "C" fn(iter: *mut NpyIter,
                                                                 outcoords: *mut npy_intp)>;

pub type PyDataMem_EventHookFunc = Option<unsafe extern "C" fn(inp: *mut c_void,
                                                               outp: *mut c_void,
                                                               size: usize,
                                                               user_data: *mut c_void)>;

pub type npy_iter_get_dataptr_t = Option<unsafe extern "C" fn(iter: *mut PyArrayIterObject,
                                                              arg1: *mut npy_intp)
                                                              -> *mut c_char>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayIterObject {
    pub ob_base: PyObject,
    pub nd_m1: c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut c_char,
    pub contiguous: npy_bool,
    pub bounds: [[npy_intp; 2usize]; 32usize],
    pub limits: [[npy_intp; 2usize]; 32usize],
    pub limits_sizes: [npy_intp; 32usize],
    pub translate: npy_iter_get_dataptr_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayMultiIterObject {
    pub ob_base: PyObject,
    pub numiter: c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: c_int,
    pub dimensions: [npy_intp; 32usize],
    pub iters: [*mut PyArrayIterObject; 32usize],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayNeighborhoodIterObject {
    pub ob_base: PyObject,
    pub nd_m1: c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut c_char,
    pub contiguous: npy_bool,
    pub bounds: [[npy_intp; 2usize]; 32usize],
    pub limits: [[npy_intp; 2usize]; 32usize],
    pub limits_sizes: [npy_intp; 32usize],
    pub translate: npy_iter_get_dataptr_t,
    pub nd: npy_intp,
    pub dimensions: [npy_intp; 32usize],
    pub _internal_iter: *mut PyArrayIterObject,
    pub constant: *mut c_char,
    pub mode: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayMapIterObject {
    pub ob_base: PyObject,
    pub numiter: c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: c_int,
    pub dimensions: [npy_intp; 32usize],
    pub outer: *mut NpyIter,
    pub unused: [*mut c_void; 30usize],
    pub array: *mut PyArrayObject,
    pub ait: *mut PyArrayIterObject,
    pub subspace: *mut PyArrayObject,
    pub iteraxes: [c_int; 32usize],
    pub fancy_strides: [npy_intp; 32usize],
    pub baseoffset: *mut c_char,
    pub consec: c_int,
    pub dataptr: *mut c_char,
    pub nd_fancy: c_int,
    pub fancy_dims: [npy_intp; 32usize],
    pub needs_api: c_int,
    pub extra_op: *mut PyArrayObject,
    pub extra_op_dtype: *mut PyArray_Descr,
    pub extra_op_flags: *mut npy_uint32,
    pub extra_op_iter: *mut NpyIter,
    pub extra_op_next: NpyIter_IterNextFunc,
    pub extra_op_ptrs: *mut *mut c_char,
    pub outer_next: NpyIter_IterNextFunc,
    pub outer_ptrs: *mut *mut c_char,
    pub outer_strides: *mut npy_intp,
    pub subspace_iter: *mut NpyIter,
    pub subspace_next: NpyIter_IterNextFunc,
    pub subspace_ptrs: *mut *mut c_char,
    pub subspace_strides: *mut npy_intp,
    pub iter_count: npy_intp,
}

extern "C" {
    pub fn NpyIter_New(op: *mut PyArrayObject,
                       flags: npy_uint32,
                       order: NPY_ORDER,
                       casting: NPY_CASTING,
                       dtype: *mut PyArray_Descr)
                       -> *mut NpyIter;
    pub fn NpyIter_MultiNew(nop: c_int,
                            op_in: *mut *mut PyArrayObject,
                            flags: npy_uint32,
                            order: NPY_ORDER,
                            casting: NPY_CASTING,
                            op_flags: *mut npy_uint32,
                            op_request_dtypes: *mut *mut PyArray_Descr)
                            -> *mut NpyIter;
    pub fn NpyIter_AdvancedNew(nop: c_int,
                               op_in: *mut *mut PyArrayObject,
                               flags: npy_uint32,
                               order: NPY_ORDER,
                               casting: NPY_CASTING,
                               op_flags: *mut npy_uint32,
                               op_request_dtypes: *mut *mut PyArray_Descr,
                               oa_ndim: c_int,
                               op_axes: *mut *mut c_int,
                               itershape: *mut npy_intp,
                               buffersize: npy_intp)
                               -> *mut NpyIter;
    pub fn NpyIter_Copy(iter: *mut NpyIter) -> *mut NpyIter;
    pub fn NpyIter_Deallocate(iter: *mut NpyIter) -> c_int;
    pub fn NpyIter_HasDelayedBufAlloc(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_HasExternalLoop(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_EnableExternalLoop(iter: *mut NpyIter) -> c_int;
    pub fn NpyIter_GetInnerStrideArray(iter: *mut NpyIter) -> *mut npy_intp;
    pub fn NpyIter_GetInnerLoopSizePtr(iter: *mut NpyIter) -> *mut npy_intp;
    pub fn NpyIter_Reset(iter: *mut NpyIter, errmsg: *mut *mut c_char) -> c_int;
    pub fn NpyIter_ResetBasePointers(iter: *mut NpyIter,
                                     baseptrs: *mut *mut c_char,
                                     errmsg: *mut *mut c_char)
                                     -> c_int;
    pub fn NpyIter_ResetToIterIndexRange(iter: *mut NpyIter,
                                         istart: npy_intp,
                                         iend: npy_intp,
                                         errmsg: *mut *mut c_char)
                                         -> c_int;
    pub fn NpyIter_GetNDim(iter: *mut NpyIter) -> c_int;
    pub fn NpyIter_GetNOp(iter: *mut NpyIter) -> c_int;
    pub fn NpyIter_GetIterNext(iter: *mut NpyIter,
                               errmsg: *mut *mut c_char)
                               -> NpyIter_IterNextFunc;
    pub fn NpyIter_GetIterSize(iter: *mut NpyIter) -> npy_intp;
    pub fn NpyIter_GetIterIndexRange(iter: *mut NpyIter,
                                     istart: *mut npy_intp,
                                     iend: *mut npy_intp);
    pub fn NpyIter_GetIterIndex(iter: *mut NpyIter) -> npy_intp;
    pub fn NpyIter_GotoIterIndex(iter: *mut NpyIter, iterindex: npy_intp) -> c_int;
    pub fn NpyIter_HasMultiIndex(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_GetShape(iter: *mut NpyIter, outshape: *mut npy_intp) -> c_int;
    pub fn NpyIter_GetGetMultiIndex(iter: *mut NpyIter,
                                    errmsg: *mut *mut c_char)
                                    -> NpyIter_GetMultiIndexFunc;
    pub fn NpyIter_GotoMultiIndex(iter: *mut NpyIter, multi_index: *mut npy_intp) -> c_int;
    pub fn NpyIter_RemoveMultiIndex(iter: *mut NpyIter) -> c_int;
    pub fn NpyIter_HasIndex(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_IsBuffered(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_IsGrowInner(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_GetBufferSize(iter: *mut NpyIter) -> npy_intp;
    pub fn NpyIter_GetIndexPtr(iter: *mut NpyIter) -> *mut npy_intp;
    pub fn NpyIter_GotoIndex(iter: *mut NpyIter, flat_index: npy_intp) -> c_int;
    pub fn NpyIter_GetDataPtrArray(iter: *mut NpyIter) -> *mut *mut c_char;
    pub fn NpyIter_GetDescrArray(iter: *mut NpyIter) -> *mut *mut PyArray_Descr;
    pub fn NpyIter_GetOperandArray(iter: *mut NpyIter) -> *mut *mut PyArrayObject;
    pub fn NpyIter_GetIterView(iter: *mut NpyIter, i: npy_intp) -> *mut PyArrayObject;
    pub fn NpyIter_GetReadFlags(iter: *mut NpyIter, outreadflags: *mut c_char);
    pub fn NpyIter_GetWriteFlags(iter: *mut NpyIter, outwriteflags: *mut c_char);
    pub fn NpyIter_DebugPrint(iter: *mut NpyIter);
    pub fn NpyIter_IterationNeedsAPI(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_GetInnerFixedStrideArray(iter: *mut NpyIter, out_strides: *mut npy_intp);
    pub fn NpyIter_RemoveAxis(iter: *mut NpyIter, axis: c_int) -> c_int;
    pub fn NpyIter_GetAxisStrideArray(iter: *mut NpyIter, axis: c_int) -> *mut npy_intp;
    pub fn NpyIter_RequiresBuffering(iter: *mut NpyIter) -> npy_bool;
    pub fn NpyIter_GetInitialDataPtrArray(iter: *mut NpyIter) -> *mut *mut c_char;
    pub fn NpyIter_CreateCompatibleStrides(iter: *mut NpyIter,
                                           itemsize: npy_intp,
                                           outstrides: *mut npy_intp)
                                           -> c_int;
    pub fn NpyIter_IsFirstVisit(iter: *mut NpyIter, iop: c_int) -> npy_bool;
}

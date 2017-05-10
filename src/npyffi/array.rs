#![allow(non_camel_case_types, non_snake_case)]

use libc::FILE;
use pyffi::*;
use std::os::raw::*;
use std::option::Option;

use super::types::*;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrayObject {
    pub ob_base: PyObject,
    pub data: *mut c_char,
    pub nd: c_int,
    pub dimensions: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub base: *mut PyObject,
    pub descr: *mut PyArray_Descr,
    pub flags: c_int,
    pub weakreflist: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_Descr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub flags: c_char,
    pub type_num: c_int,
    pub elsize: c_int,
    pub alignment: c_int,
    pub subarray: *mut PyArrray_ArrayDescr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub f: *mut PyArray_ArrFuncs,
    pub metadata: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
    pub hash: npy_hash_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrray_ArrayDescr {
    pub base: *mut PyArray_Descr,
    pub shape: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_ArrFuncs {
    pub cast: [PyArray_VectorUnaryFunc; 21usize],
    pub getitem: PyArray_GetItemFunc,
    pub setitem: PyArray_SetItemFunc,
    pub copyswapn: PyArray_CopySwapNFunc,
    pub copyswap: PyArray_CopySwapFunc,
    pub compare: PyArray_CompareFunc,
    pub argmax: PyArray_ArgFunc,
    pub dotfunc: PyArray_DotFunc,
    pub scanfunc: PyArray_ScanFunc,
    pub fromstr: PyArray_FromStrFunc,
    pub nonzero: PyArray_NonzeroFunc,
    pub fill: PyArray_FillFunc,
    pub fillwithscalar: PyArray_FillWithScalarFunc,
    pub sort: [PyArray_SortFunc; 3usize],
    pub argsort: [PyArray_ArgSortFunc; 3usize],
    pub castdict: *mut PyObject,
    pub scalarkind: PyArray_ScalarKindFunc,
    pub cancastscalarkindto: *mut *mut c_int,
    pub cancastto: *mut c_int,
    pub fastclip: PyArray_FastClipFunc,
    pub fastputmask: PyArray_FastPutmaskFunc,
    pub fasttake: PyArray_FastTakeFunc,
    pub argmin: PyArray_ArgFunc,
}

pub type PyArray_GetItemFunc = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)
                                                           -> *mut PyObject>;
pub type PyArray_SetItemFunc = Option<unsafe extern "C" fn(*mut PyObject,
                                                           *mut c_void,
                                                           *mut c_void)
                                                           -> c_int>;
pub type PyArray_CopySwapNFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                             npy_intp,
                                                             *mut c_void,
                                                             npy_intp,
                                                             npy_intp,
                                                             c_int,
                                                             *mut c_void)>;
pub type PyArray_CopySwapFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                            *mut c_void,
                                                            c_int,
                                                            *mut c_void)>;
pub type PyArray_NonzeroFunc = Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uchar>;
pub type PyArray_CompareFunc = Option<unsafe extern "C" fn(*const c_void,
                                                           *const c_void,
                                                           *mut c_void)
                                                           -> c_int>;
pub type PyArray_ArgFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                       npy_intp,
                                                       *mut npy_intp,
                                                       *mut c_void)
                                                       -> c_int>;
pub type PyArray_DotFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                       npy_intp,
                                                       *mut c_void,
                                                       npy_intp,
                                                       *mut c_void,
                                                       npy_intp,
                                                       *mut c_void)>;
pub type PyArray_VectorUnaryFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                               *mut c_void,
                                                               npy_intp,
                                                               *mut c_void,
                                                               *mut c_void)>;
pub type PyArray_ScanFunc = Option<unsafe extern "C" fn(*mut FILE,
                                                        *mut c_void,
                                                        *mut c_char,
                                                        *mut PyArray_Descr)
                                                        -> c_int>;
pub type PyArray_FromStrFunc = Option<unsafe extern "C" fn(*mut c_char,
                                                           *mut c_void,
                                                           *mut *mut c_char,
                                                           *mut PyArray_Descr)
                                                           -> c_int>;
pub type PyArray_FillFunc = Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void) -> c_int>;
pub type PyArray_SortFunc = Option<unsafe extern "C" fn(*mut c_void, npy_intp, *mut c_void) -> c_int>;
pub type PyArray_ArgSortFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                           *mut npy_intp,
                                                           npy_intp,
                                                           *mut c_void)
                                                           -> c_int>;
pub type PyArray_PartitionFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                             npy_intp,
                                                             npy_intp,
                                                             *mut npy_intp,
                                                             *mut npy_intp,
                                                             *mut c_void)
                                                             -> c_int>;
pub type PyArray_ArgPartitionFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                                *mut npy_intp,
                                                                npy_intp,
                                                                npy_intp,
                                                                *mut npy_intp,
                                                                *mut npy_intp,
                                                                *mut c_void)
                                                                -> c_int>;
pub type PyArray_FillWithScalarFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                                  npy_intp,
                                                                  *mut c_void,
                                                                  *mut c_void)
                                                                  -> c_int>;
pub type PyArray_ScalarKindFunc = Option<unsafe extern "C" fn(*mut c_void) -> c_int>;
pub type PyArray_FastClipFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                            npy_intp,
                                                            *mut c_void,
                                                            *mut c_void,
                                                            *mut c_void)>;
pub type PyArray_FastPutmaskFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                               *mut c_void,
                                                               npy_intp,
                                                               *mut c_void,
                                                               npy_intp)>;
pub type PyArray_FastTakeFunc = Option<unsafe extern "C" fn(*mut c_void,
                                                            *mut c_void,
                                                            *mut npy_intp,
                                                            npy_intp,
                                                            npy_intp,
                                                            npy_intp,
                                                            npy_intp,
                                                            NPY_CLIPMODE)
                                                            -> c_int>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayFlagsObject {
    pub ob_base: PyObject,
    pub arr: *mut PyObject,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Dims {
    pub ptr: *mut npy_intp,
    pub len: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Chunk {
    pub ob_base: PyObject,
    pub base: *mut PyObject,
    pub ptr: *mut c_void,
    pub len: npy_intp,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayInterface {
    pub two: c_int,
    pub nd: c_int,
    pub typekind: c_char,
    pub itemsize: c_int,
    pub flags: c_int,
    pub shape: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub data: *mut c_void,
    pub descr: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut c_void; 2usize],
}

pub type NpyAuxData_FreeFunc = Option<unsafe extern "C" fn(*mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc = Option<unsafe extern "C" fn(*mut NpyAuxData) -> *mut NpyAuxData>;

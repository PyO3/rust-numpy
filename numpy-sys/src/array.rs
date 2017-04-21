#![allow(non_camel_case_types)]

use libc::FILE;
use pyffi::*;
use super::types::*;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_CLIPMODE {
    NPY_CLIP = 0,
    NPY_WRAP = 1,
    NPY_RAISE = 2,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct _object {
    pub ob_refcnt: Py_ssize_t,
    pub ob_type: *mut PyTypeObject,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayFlagsObject {
    pub ob_base: PyObject,
    pub arr: *mut PyObject,
    pub flags: ::std::os::raw::c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Dims {
    pub ptr: *mut npy_intp,
    pub len: ::std::os::raw::c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Chunk {
    pub ob_base: PyObject,
    pub base: *mut PyObject,
    pub ptr: *mut ::std::os::raw::c_void,
    pub len: npy_intp,
    pub flags: ::std::os::raw::c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayInterface {
    pub two: ::std::os::raw::c_int,
    pub nd: ::std::os::raw::c_int,
    pub typekind: ::std::os::raw::c_char,
    pub itemsize: ::std::os::raw::c_int,
    pub flags: ::std::os::raw::c_int,
    pub shape: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub data: *mut ::std::os::raw::c_void,
    pub descr: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_Descr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: ::std::os::raw::c_char,
    pub type_: ::std::os::raw::c_char,
    pub byteorder: ::std::os::raw::c_char,
    pub flags: ::std::os::raw::c_char,
    pub type_num: ::std::os::raw::c_int,
    pub elsize: ::std::os::raw::c_int,
    pub alignment: ::std::os::raw::c_int,
    pub subarray: *mut _PyArray_Descr__arr_descr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub f: *mut PyArray_ArrFuncs,
    pub metadata: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
    pub hash: npy_hash_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrayObject {
    pub ob_base: PyObject,
    pub data: *mut ::std::os::raw::c_char,
    pub nd: ::std::os::raw::c_int,
    pub dimensions: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub base: *mut PyObject,
    pub descr: *mut PyArray_Descr,
    pub flags: ::std::os::raw::c_int,
    pub weakreflist: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct _PyArray_Descr__arr_descr {
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
    pub cancastscalarkindto: *mut *mut ::std::os::raw::c_int,
    pub cancastto: *mut ::std::os::raw::c_int,
    pub fastclip: PyArray_FastClipFunc,
    pub fastputmask: PyArray_FastPutmaskFunc,
    pub fasttake: PyArray_FastTakeFunc,
    pub argmin: PyArray_ArgFunc,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut ::std::os::raw::c_void; 2usize],
}


pub type PyArray_GetItemFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void)
                                                 -> *mut _object>;
pub type PyArray_SetItemFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut PyObject,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_CopySwapNFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: npy_intp,
                                                 arg5: npy_intp,
                                                 arg6: ::std::os::raw::c_int,
                                                 arg7: *mut ::std::os::raw::c_void)>;
pub type PyArray_CopySwapFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: ::std::os::raw::c_int,
                                                 arg4: *mut ::std::os::raw::c_void)>;
pub type PyArray_NonzeroFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_uchar>;
pub type PyArray_CompareFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                                 arg2: *const ::std::os::raw::c_void,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_DotFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: npy_intp,
                                                 arg5: *mut ::std::os::raw::c_void,
                                                 arg6: npy_intp,
                                                 arg7: *mut ::std::os::raw::c_void)>;
pub type PyArray_VectorUnaryFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void,
                                                 arg5: *mut ::std::os::raw::c_void)>;
pub type PyArray_ScanFunc =
    ::std::option::Option<unsafe extern "C" fn(fp: *mut FILE,
                                                 dptr: *mut ::std::os::raw::c_void,
                                                 ignore: *mut ::std::os::raw::c_char,
                                                 arg1: *mut PyArray_Descr)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FromStrFunc =
    ::std::option::Option<unsafe extern "C" fn(s: *mut ::std::os::raw::c_char,
                                                 dptr: *mut ::std::os::raw::c_void,
                                                 endptr: *mut *mut ::std::os::raw::c_char,
                                                 arg1: *mut PyArray_Descr)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FillFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_SortFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgSortFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_PartitionFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: *mut npy_intp,
                                                 arg5: *mut npy_intp,
                                                 arg6: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgPartitionFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: npy_intp,
                                                 arg5: *mut npy_intp,
                                                 arg6: *mut npy_intp,
                                                 arg7: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FillWithScalarFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ScalarKindFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FastClipFunc =
    ::std::option::Option<unsafe extern "C" fn(in_: *mut ::std::os::raw::c_void,
                                                 n_in: npy_intp,
                                                 min: *mut ::std::os::raw::c_void,
                                                 max: *mut ::std::os::raw::c_void,
                                                 out: *mut ::std::os::raw::c_void)>;
pub type PyArray_FastPutmaskFunc =
    ::std::option::Option<unsafe extern "C" fn(in_: *mut ::std::os::raw::c_void,
                                                 mask: *mut ::std::os::raw::c_void,
                                                 n_in: npy_intp,
                                                 values: *mut ::std::os::raw::c_void,
                                                 nv: npy_intp)>;
pub type PyArray_FastTakeFunc =
    ::std::option::Option<unsafe extern "C" fn(dest: *mut ::std::os::raw::c_void,
                                                 src: *mut ::std::os::raw::c_void,
                                                 indarray: *mut npy_intp,
                                                 nindarray: npy_intp,
                                                 n_outer: npy_intp,
                                                 m_middle: npy_intp,
                                                 nelem: npy_intp,
                                                 clipmode: NPY_CLIPMODE)
                                                 -> ::std::os::raw::c_int>;
pub type NpyAuxData_FreeFunc = ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData) -> *mut NpyAuxData>;

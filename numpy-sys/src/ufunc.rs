#![allow(non_camel_case_types)]

use pyffi::*;
use std::os::raw::*;
use std::option::Option;

use super::types::*;
use super::array::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyUFuncObject {
    pub ob_base: PyObject,
    pub nin: c_int,
    pub nout: c_int,
    pub nargs: c_int,
    pub identity: c_int,
    pub functions: *mut PyUFuncGenericFunction,
    pub data: *mut *mut c_void,
    pub ntypes: c_int,
    pub reserved1: c_int,
    pub name: *const c_char,
    pub types: *mut c_char,
    pub doc: *const c_char,
    pub ptr: *mut c_void,
    pub obj: *mut PyObject,
    pub userloops: *mut PyObject,
    pub core_enabled: c_int,
    pub core_num_dim_ix: c_int,
    pub core_num_dims: *mut c_int,
    pub core_dim_ixs: *mut c_int,
    pub core_offsets: *mut c_int,
    pub core_signature: *mut c_char,
    pub type_resolver: PyUFunc_TypeResolutionFunc,
    pub legacy_inner_loop_selector: PyUFunc_LegacyInnerLoopSelectionFunc,
    pub reserved2: *mut c_void,
    pub masked_inner_loop_selector: PyUFunc_MaskedInnerLoopSelectionFunc,
    pub op_flags: *mut npy_uint32,
    pub iter_flags: npy_uint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut c_void; 2usize],
}

pub type NpyAuxData_FreeFunc = Option<unsafe extern "C" fn(arg1: *mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc = Option<unsafe extern "C" fn(arg1: *mut NpyAuxData) -> *mut NpyAuxData>;

pub type PyUFuncGenericFunction = Option<unsafe extern "C" fn(args: *mut *mut c_char,
                                                              dimensions: *mut npy_intp,
                                                              strides: *mut npy_intp,
                                                              innerloopdata: *mut c_void)>;
pub type PyUFunc_MaskedStridedInnerLoopFunc = Option<unsafe extern "C" fn(dataptrs: *mut *mut c_char,
                                                                          strides: *mut npy_intp,
                                                                          maskptr: *mut c_char,
                                                                          mask_stride: npy_intp,
                                                                          count: npy_intp,
                                                                          innerloopdata: *mut NpyAuxData)>;
pub type PyUFunc_TypeResolutionFunc = Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                                                  casting: NPY_CASTING,
                                                                  operands: *mut *mut PyArrayObject,
                                                                  type_tup: *mut PyObject,
                                                                  out_dtypes: *mut *mut PyArray_Descr)
                                                                  -> c_int>;
pub type PyUFunc_LegacyInnerLoopSelectionFunc =
    Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                dtypes: *mut *mut PyArray_Descr,
                                out_innerloop: *mut PyUFuncGenericFunction,
                                out_innerloopdata: *mut *mut c_void,
                                out_needs_api: *mut c_int)
                                -> c_int>;
pub type PyUFunc_MaskedInnerLoopSelectionFunc =
    Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                dtypes: *mut *mut PyArray_Descr,
                                mask_dtype: *mut PyArray_Descr,
                                fixed_strides: *mut npy_intp,
                                fixed_mask_stride: npy_intp,
                                out_innerloop: *mut PyUFunc_MaskedStridedInnerLoopFunc,
                                out_innerloopdata: *mut *mut NpyAuxData,
                                out_needs_api: *mut c_int)
                                -> c_int>;

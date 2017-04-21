#![allow(non_camel_case_types)]
// FIXME ^ should be removed

use pyffi::*;
use super::types::*;
use super::array::*;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_CASTING {
    NPY_NO_CASTING = 0,
    NPY_EQUIV_CASTING = 1,
    NPY_SAFE_CASTING = 2,
    NPY_SAME_KIND_CASTING = 3,
    NPY_UNSAFE_CASTING = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyUFuncObject {
    pub ob_base: PyObject,
    pub nin: ::std::os::raw::c_int,
    pub nout: ::std::os::raw::c_int,
    pub nargs: ::std::os::raw::c_int,
    pub identity: ::std::os::raw::c_int,
    pub functions: *mut PyUFuncGenericFunction,
    pub data: *mut *mut ::std::os::raw::c_void,
    pub ntypes: ::std::os::raw::c_int,
    pub reserved1: ::std::os::raw::c_int,
    pub name: *const ::std::os::raw::c_char,
    pub types: *mut ::std::os::raw::c_char,
    pub doc: *const ::std::os::raw::c_char,
    pub ptr: *mut ::std::os::raw::c_void,
    pub obj: *mut PyObject,
    pub userloops: *mut PyObject,
    pub core_enabled: ::std::os::raw::c_int,
    pub core_num_dim_ix: ::std::os::raw::c_int,
    pub core_num_dims: *mut ::std::os::raw::c_int,
    pub core_dim_ixs: *mut ::std::os::raw::c_int,
    pub core_offsets: *mut ::std::os::raw::c_int,
    pub core_signature: *mut ::std::os::raw::c_char,
    pub type_resolver: PyUFunc_TypeResolutionFunc,
    pub legacy_inner_loop_selector: PyUFunc_LegacyInnerLoopSelectionFunc,
    pub reserved2: *mut ::std::os::raw::c_void,
    pub masked_inner_loop_selector: PyUFunc_MaskedInnerLoopSelectionFunc,
    pub op_flags: *mut npy_uint32,
    pub iter_flags: npy_uint32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut ::std::os::raw::c_void; 2usize],
}

pub type NpyAuxData_FreeFunc = ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData) -> *mut NpyAuxData>;

pub type PyUFuncGenericFunction =
    ::std::option::Option<unsafe extern "C" fn(args: *mut *mut ::std::os::raw::c_char,
                                                 dimensions: *mut npy_intp,
                                                 strides: *mut npy_intp,
                                                 innerloopdata: *mut ::std::os::raw::c_void)>;
pub type PyUFunc_MaskedStridedInnerLoopFunc =
    ::std::option::Option<unsafe extern "C" fn(dataptrs: *mut *mut ::std::os::raw::c_char,
                                                 strides: *mut npy_intp,
                                                 maskptr: *mut ::std::os::raw::c_char,
                                                 mask_stride: npy_intp,
                                                 count: npy_intp,
                                                 innerloopdata: *mut NpyAuxData)>;
pub type PyUFunc_TypeResolutionFunc =
    ::std::option::Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                                 casting: NPY_CASTING,
                                                 operands: *mut *mut PyArrayObject,
                                                 type_tup: *mut PyObject,
                                                 out_dtypes: *mut *mut PyArray_Descr)
                                                 -> ::std::os::raw::c_int>;
pub type PyUFunc_LegacyInnerLoopSelectionFunc =
    ::std::option::Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                               dtypes:
                                                   *mut *mut PyArray_Descr,
                                               out_innerloop:
                                                   *mut PyUFuncGenericFunction,
                                               out_innerloopdata:
                                                   *mut *mut ::std::os::raw::c_void,
                                               out_needs_api:
                                                   *mut ::std::os::raw::c_int)
                              -> ::std::os::raw::c_int>;
pub type PyUFunc_MaskedInnerLoopSelectionFunc =
    ::std::option::Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                               dtypes:
                                                   *mut *mut PyArray_Descr,
                                               mask_dtype: *mut PyArray_Descr,
                                               fixed_strides: *mut npy_intp,
                                               fixed_mask_stride: npy_intp,
                                               out_innerloop:
                                                   *mut PyUFunc_MaskedStridedInnerLoopFunc,
                                               out_innerloopdata:
                                                   *mut *mut NpyAuxData,
                                               out_needs_api:
                                                   *mut ::std::os::raw::c_int)
                              -> ::std::os::raw::c_int>;

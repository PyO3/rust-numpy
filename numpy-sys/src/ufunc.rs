#![allow(non_camel_case_types)]

use pyffi::*;
use std::os::raw::*;
use std::option::Option;

use super::types::*;
use super::array::*;

extern "C" {
    pub static PyUFunc_Type: PyTypeObject;
}

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
pub type PyUFunc_LegacyInnerLoopSelectionFunc = Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                                                            dtypes: *mut *mut PyArray_Descr,
                                                                            out_innerloop: *mut PyUFuncGenericFunction,
                                                                            out_innerloopdata: *mut *mut c_void,
                                                                            out_needs_api: *mut c_int)
                                                                            -> c_int>;
pub type PyUFunc_MaskedInnerLoopSelectionFunc = Option<unsafe extern "C" fn(ufunc: *mut PyUFuncObject,
                                                                            dtypes: *mut *mut PyArray_Descr,
                                                                            mask_dtype: *mut PyArray_Descr,
                                                                            fixed_strides: *mut npy_intp,
                                                                            fixed_mask_stride: npy_intp,
                                                                            out_innerloop: *mut PyUFunc_MaskedStridedInnerLoopFunc,
                                                                            out_innerloopdata: *mut *mut NpyAuxData,
                                                                            out_needs_api: *mut c_int)
                                                                            -> c_int>;

extern "C" {
    static PyUFunc_API: *const c_void;
}

macro_rules! pyufunc_api {
    [ $offset:expr; $fname:ident ( $($arg:ident : $t:ty),* ) $( -> $ret:ty )* ] => {

#[allow(non_snake_case)]
pub unsafe fn $fname($($arg : $t), *) $( -> $ret )* {
    let api = &PyUFunc_API as *const *const c_void;
    let fptr = api.offset($offset) as (*const extern fn ($($arg : $t), *) $( -> $ret )* );
    (*fptr)($($arg), *)
}

}} // pyufunc_api!

pyufunc_api![1; PyUFunc_FromFuncAndData(func: *mut PyUFuncGenericFunction, data: *mut *mut c_void, types: *mut c_char, ntypes: c_int, nin: c_int, nout: c_int, identity: c_int, name: *const c_char, doc: *const c_char, unused: c_int) -> *mut PyObject];
pyufunc_api![2; PyUFunc_RegisterLoopForType(ufunc: *mut PyUFuncObject, usertype: c_int, function: PyUFuncGenericFunction, arg_types: *mut c_int, data: *mut c_void) -> c_int];
pyufunc_api![3; PyUFunc_GenericFunction(ufunc: *mut PyUFuncObject, args: *mut PyObject, kwds: *mut PyObject, op: *mut *mut PyArrayObject) -> c_int];
pyufunc_api![4; PyUFunc_f_f_As_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![5; PyUFunc_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![6; PyUFunc_f_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![7; PyUFunc_g_g(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![8; PyUFunc_F_F_As_D_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![9; PyUFunc_F_F(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![10; PyUFunc_D_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![11; PyUFunc_G_G(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![12; PyUFunc_O_O(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![13; PyUFunc_ff_f_As_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![14; PyUFunc_ff_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![15; PyUFunc_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![16; PyUFunc_gg_g(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![17; PyUFunc_FF_F_As_DD_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![18; PyUFunc_DD_D(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![19; PyUFunc_FF_F(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![20; PyUFunc_GG_G(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![21; PyUFunc_OO_O(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![22; PyUFunc_O_O_method(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![23; PyUFunc_OO_O_method(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![24; PyUFunc_On_Om(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![25; PyUFunc_GetPyValues(name: *mut c_char, bufsize: *mut c_int, errmask: *mut c_int, errobj: *mut *mut PyObject) -> c_int];
pyufunc_api![26; PyUFunc_checkfperr(errmask: c_int, errobj: *mut PyObject, first: *mut c_int) -> c_int];
pyufunc_api![27; PyUFunc_clearfperr()];
pyufunc_api![28; PyUFunc_getfperr() -> c_int];
pyufunc_api![29; PyUFunc_handlefperr(errmask: c_int, errobj: *mut PyObject, retstatus: c_int, first: *mut c_int) -> c_int];
pyufunc_api![30; PyUFunc_ReplaceLoopBySignature(func: *mut PyUFuncObject, newfunc: PyUFuncGenericFunction, signature: *mut c_int, oldfunc: *mut PyUFuncGenericFunction) -> c_int];
pyufunc_api![31; PyUFunc_FromFuncAndDataAndSignature(func: *mut PyUFuncGenericFunction, data: *mut *mut c_void, types: *mut c_char, ntypes: c_int, nin: c_int, nout: c_int, identity: c_int, name: *const c_char, doc: *const c_char, unused: c_int, signature: *const c_char) -> *mut PyObject];
pyufunc_api![32; PyUFunc_SetUsesArraysAsData(data: *mut *mut c_void, i: usize) -> c_int];
pyufunc_api![33; PyUFunc_e_e(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![34; PyUFunc_e_e_As_f_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![35; PyUFunc_e_e_As_d_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![36; PyUFunc_ee_e(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![37; PyUFunc_ee_e_As_ff_f(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![38; PyUFunc_ee_e_As_dd_d(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, func: *mut c_void)];
pyufunc_api![39; PyUFunc_DefaultTypeResolver(ufunc: *mut PyUFuncObject, casting: NPY_CASTING, operands: *mut *mut PyArrayObject, type_tup: *mut PyObject, out_dtypes: *mut *mut PyArray_Descr) -> c_int];
pyufunc_api![40; PyUFunc_ValidateCasting(ufunc: *mut PyUFuncObject, casting: NPY_CASTING, operands: *mut *mut PyArrayObject, dtypes: *mut *mut PyArray_Descr) -> c_int];
pyufunc_api![41; PyUFunc_RegisterLoopForDescr(ufunc: *mut PyUFuncObject, user_dtype: *mut PyArray_Descr, function: PyUFuncGenericFunction, arg_dtypes: *mut *mut PyArray_Descr, data: *mut c_void) -> c_int];

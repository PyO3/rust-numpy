#![allow(non_camel_case_types)]
// FIXME ^ should be removed

use pyffi::*;
use super::types::*;
use super::array::*;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NpyIter([u8; 0]);

pub type NpyIter_IterNextFunc =
    ::std::option::Option<unsafe extern "C" fn(iter: *mut NpyIter) -> ::std::os::raw::c_int>;

pub type NpyIter_GetMultiIndexFunc =
    ::std::option::Option<unsafe extern "C" fn(iter: *mut NpyIter, outcoords: *mut npy_intp)>;

pub type PyDataMem_EventHookFunc =
    ::std::option::Option<unsafe extern "C" fn(inp: *mut ::std::os::raw::c_void,
                                                 outp: *mut ::std::os::raw::c_void,
                                                 size: usize,
                                                 user_data: *mut ::std::os::raw::c_void)>;

pub type npy_iter_get_dataptr_t =
    ::std::option::Option<unsafe extern "C" fn(iter: *mut PyArrayIterObject,
                                                 arg1: *mut npy_intp)
                                                 -> *mut ::std::os::raw::c_char>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayIterObject {
    pub ob_base: PyObject,
    pub nd_m1: ::std::os::raw::c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut ::std::os::raw::c_char,
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
    pub numiter: ::std::os::raw::c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: ::std::os::raw::c_int,
    pub dimensions: [npy_intp; 32usize],
    pub iters: [*mut PyArrayIterObject; 32usize],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayNeighborhoodIterObject {
    pub ob_base: PyObject,
    pub nd_m1: ::std::os::raw::c_int,
    pub index: npy_intp,
    pub size: npy_intp,
    pub coordinates: [npy_intp; 32usize],
    pub dims_m1: [npy_intp; 32usize],
    pub strides: [npy_intp; 32usize],
    pub backstrides: [npy_intp; 32usize],
    pub factors: [npy_intp; 32usize],
    pub ao: *mut PyArrayObject,
    pub dataptr: *mut ::std::os::raw::c_char,
    pub contiguous: npy_bool,
    pub bounds: [[npy_intp; 2usize]; 32usize],
    pub limits: [[npy_intp; 2usize]; 32usize],
    pub limits_sizes: [npy_intp; 32usize],
    pub translate: npy_iter_get_dataptr_t,
    pub nd: npy_intp,
    pub dimensions: [npy_intp; 32usize],
    pub _internal_iter: *mut PyArrayIterObject,
    pub constant: *mut ::std::os::raw::c_char,
    pub mode: ::std::os::raw::c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayMapIterObject {
    pub ob_base: PyObject,
    pub numiter: ::std::os::raw::c_int,
    pub size: npy_intp,
    pub index: npy_intp,
    pub nd: ::std::os::raw::c_int,
    pub dimensions: [npy_intp; 32usize],
    pub outer: *mut NpyIter,
    pub unused: [*mut ::std::os::raw::c_void; 30usize],
    pub array: *mut PyArrayObject,
    pub ait: *mut PyArrayIterObject,
    pub subspace: *mut PyArrayObject,
    pub iteraxes: [::std::os::raw::c_int; 32usize],
    pub fancy_strides: [npy_intp; 32usize],
    pub baseoffset: *mut ::std::os::raw::c_char,
    pub consec: ::std::os::raw::c_int,
    pub dataptr: *mut ::std::os::raw::c_char,
    pub nd_fancy: ::std::os::raw::c_int,
    pub fancy_dims: [npy_intp; 32usize],
    pub needs_api: ::std::os::raw::c_int,
    pub extra_op: *mut PyArrayObject,
    pub extra_op_dtype: *mut PyArray_Descr,
    pub extra_op_flags: *mut npy_uint32,
    pub extra_op_iter: *mut NpyIter,
    pub extra_op_next: NpyIter_IterNextFunc,
    pub extra_op_ptrs: *mut *mut ::std::os::raw::c_char,
    pub outer_next: NpyIter_IterNextFunc,
    pub outer_ptrs: *mut *mut ::std::os::raw::c_char,
    pub outer_strides: *mut npy_intp,
    pub subspace_iter: *mut NpyIter,
    pub subspace_next: NpyIter_IterNextFunc,
    pub subspace_ptrs: *mut *mut ::std::os::raw::c_char,
    pub subspace_strides: *mut npy_intp,
    pub iter_count: npy_intp,
}

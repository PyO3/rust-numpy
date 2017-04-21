#![allow(non_camel_case_types)]

use pyffi::{Py_intptr_t, Py_uintptr_t, Py_hash_t};

pub type npy_intp = Py_intptr_t;
pub type npy_uintp = Py_uintptr_t;
pub type npy_longlong = ::std::os::raw::c_longlong;
pub type npy_ulonglong = ::std::os::raw::c_ulonglong;
pub type npy_bool = ::std::os::raw::c_uchar;
pub type npy_longdouble = f64;
pub type npy_byte = ::std::os::raw::c_char;
pub type npy_ubyte = ::std::os::raw::c_uchar;
pub type npy_ushort = ::std::os::raw::c_ushort;
pub type npy_uint = ::std::os::raw::c_uint;
pub type npy_ulong = ::std::os::raw::c_ulong;
pub type npy_char = ::std::os::raw::c_char;
pub type npy_short = ::std::os::raw::c_short;
pub type npy_int = ::std::os::raw::c_int;
pub type npy_long = ::std::os::raw::c_long;
pub type npy_float = f32;
pub type npy_double = f64;
pub type npy_hash_t = Py_hash_t;
pub type npy_int64 = ::std::os::raw::c_long;
pub type npy_uint64 = ::std::os::raw::c_ulong;
pub type npy_int32 = ::std::os::raw::c_int;
pub type npy_uint32 = ::std::os::raw::c_uint;
pub type npy_ucs4 = ::std::os::raw::c_uint;
pub type npy_int16 = ::std::os::raw::c_short;
pub type npy_uint16 = ::std::os::raw::c_ushort;
pub type npy_int8 = ::std::os::raw::c_char;
pub type npy_uint8 = ::std::os::raw::c_uchar;
pub type npy_float64 = f64;
pub type npy_complex128 = npy_cdouble;
pub type npy_float32 = f32;
pub type npy_complex64 = npy_cfloat;
pub type npy_half = npy_uint16;
pub type npy_float16 = npy_half;
pub type npy_float128 = npy_longdouble;
pub type npy_complex256 = npy_clongdouble;
pub type npy_timedelta = npy_int64;
pub type npy_datetime = npy_int64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_cdouble {
    pub real: f64,
    pub imag: f64,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_cfloat {
    pub real: f32,
    pub imag: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_clongdouble {
    pub real: npy_longdouble,
    pub imag: npy_longdouble,
}

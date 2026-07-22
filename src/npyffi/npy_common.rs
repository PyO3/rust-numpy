use std::ffi::c_int;

/// Unknown CPU endianness.
pub const NPY_CPU_UNKNOWN_ENDIAN: c_int = 0;
/// CPU is little-endian.
pub const NPY_CPU_LITTLE: c_int = 1;
/// CPU is big-endian.
pub const NPY_CPU_BIG: c_int = 2;

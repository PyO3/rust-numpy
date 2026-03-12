// This file matches the numpyconfig.h header.

use std::ffi::c_uint;

/// The current target ABI version
const NPY_ABI_VERSION: c_uint = 0x02000000;

/// The current target API version (v1.15)
const NPY_API_VERSION: c_uint = 0x0000000c;

pub(super) const NPY_2_0_API_VERSION: c_uint = 0x00000012;

/// The current version of the `ndarray` object (ABI version).
pub const NPY_VERSION: c_uint = NPY_ABI_VERSION;
/// The current version of C API.
pub const NPY_FEATURE_VERSION: c_uint = NPY_API_VERSION;
/// The string representation of current version C API.
pub const NPY_FEATURE_VERSION_STRING: &str = "1.15";

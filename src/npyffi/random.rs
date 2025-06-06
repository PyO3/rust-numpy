use std::{ffi::c_void, ptr::NonNull};

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyCapsule};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct npy_bitgen {
    pub state: *mut c_void,
    pub next_uint64: NonNull<unsafe extern "C" fn(*mut c_void) -> super::npy_uint64>, //nogil
    pub next_uint32: NonNull<unsafe extern "C" fn(*mut c_void) -> super::npy_uint32>, //nogil
    pub next_double: NonNull<unsafe extern "C" fn(*mut c_void) -> libc::c_double>, //nogil
    pub next_raw: NonNull<unsafe extern "C" fn(*mut c_void) -> super::npy_uint64>, //nogil
}

pub fn get_bitgen_api<'py>(bitgen: Bound<'py, PyAny>) -> PyResult<NonNull<npy_bitgen>> {
    let capsule = bitgen.getattr("capsule")?.downcast_into::<PyCapsule>()?;
    assert_eq!(capsule.name()?, Some(c"BitGenerator"));
    let ptr = capsule.pointer() as *mut npy_bitgen;
    NonNull::new(ptr).ok_or_else(|| PyRuntimeError::new_err("Invalid BitGenerator capsule"))
}

use std::ffi::c_void;


#[repr(C)]
#[derive(Debug, Clone, Copy)] // TODO: can it be Clone and/or Copy?
pub struct npy_bitgen {
    pub state: *mut c_void,
    pub next_uint64: unsafe extern "C" fn(*mut c_void) -> super::npy_uint64, //nogil
    pub next_uint32: unsafe extern "C" fn(*mut c_void) -> super::npy_uint32, //nogil
    pub next_double: unsafe extern "C" fn(*mut c_void) -> libc::c_double, //nogil
    pub next_raw: unsafe extern "C" fn(*mut c_void) -> super::npy_uint64, //nogil
}

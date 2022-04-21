//! Low-Level bindings for NumPy C API.
//!
//! <https://numpy.org/doc/stable/reference/c-api>
#![allow(
    non_camel_case_types,
    missing_docs,
    missing_debug_implementations,
    clippy::too_many_arguments,
    clippy::missing_safety_doc
)]

use pyo3::{ffi, Python};
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr::null_mut;

fn get_numpy_api(_py: Python, module: &str, capsule: &str) -> *const *const c_void {
    let module = CString::new(module).unwrap();
    let capsule = CString::new(capsule).unwrap();
    unsafe {
        let module = ffi::PyImport_ImportModule(module.as_ptr());
        assert!(!module.is_null(), "Failed to import NumPy module");
        let capsule = ffi::PyObject_GetAttrString(module as _, capsule.as_ptr());
        assert!(!capsule.is_null(), "Failed to get NumPy API capsule");
        ffi::PyCapsule_GetPointer(capsule, null_mut()) as _
    }
}

// Implements wrappers for NumPy's Array and UFunc API
macro_rules! impl_api {
    [$offset: expr; $fname: ident ( $($arg: ident : $t: ty),* $(,)?) $( -> $ret: ty )* ] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname(&self, py: Python, $($arg : $t), *) $( -> $ret )* {
            let fptr = self.get(py, $offset)
                           as *const extern fn ($($arg : $t), *) $( -> $ret )*;
            (*fptr)($($arg), *)
        }
    };
}

pub mod array;
pub mod flags;
pub mod objects;
pub mod types;
pub mod ufunc;

pub use self::array::*;
pub use self::flags::*;
pub use self::objects::*;
pub use self::types::*;
pub use self::ufunc::*;

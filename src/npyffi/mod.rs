//! Low-Level bindings for NumPy C API.
//!
//! https://docs.scipy.org/doc/numpy/reference/c-api.html
//!
//! Most of functions in this submodule are unsafe.
//! If you use functions in this submodule, you need to understand
//! basic usage of Python C API, especially for the reference counting.
//!
//! - http://docs.python.jp/3/c-api/
//! - http://dgrunwald.github.io/rust-pyo3/doc/pyo3/

use pyo3::ffi;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr::null_mut;

fn get_numpy_api(module: &str, capsule: &str) -> *const *const c_void {
    let module = CString::new(module).unwrap();
    let capsule = CString::new(capsule).unwrap();
    unsafe fn get_capsule(capsule: *mut ffi::PyObject) -> *const *const c_void {
        ffi::PyCapsule_GetPointer(capsule, null_mut()) as *const *const c_void
    }
    unsafe {
        assert_ne!(
            ffi::Py_IsInitialized(),
            0,
            r"Numpy API is called before initializing Python!
Please make sure that you get gil, by `let gil = Python::acquire_gil();`"
        );
        let numpy = ffi::PyImport_ImportModule(module.as_ptr());
        assert!(!numpy.is_null(), "Failed to import numpy module");
        let capsule = ffi::PyObject_GetAttrString(numpy as *mut ffi::PyObject, capsule.as_ptr());
        assert!(!capsule.is_null(), "Failed to import numpy module");
        get_capsule(capsule)
    }
}

// Define Array&UFunc APIs
macro_rules! impl_api {
    [ $offset:expr; $fname:ident ( $($arg:ident : $t:ty),* ) $( -> $ret:ty )* ] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname(&self, $($arg : $t), *) $( -> $ret )* {
            let fptr = self.0.offset($offset)
                               as (*const extern fn ($($arg : $t), *) $( -> $ret )* );
            (*fptr)($($arg), *)
        }
    }
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

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

use pyo3::{ffi, ObjectProtocol, Python, ToPyPointer};
use std::os::raw::c_void;
use std::ptr::null_mut;

fn get_numpy_api(module: &str, capsule: &str) -> *const *const c_void {
    let gil = Python::acquire_gil();
    let numpy = gil
        .python()
        .import("numpy.core.multiarray")
        .expect("Failed to import numpy.core.multiarray");
    let capsule = numpy
        .getattr("_ARRAY_API")
        .expect("Failed to import numpy.core.multiarray._ARRAY_API");
    unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), null_mut()) as *const *const c_void }
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
pub mod objects;
pub mod types;
pub mod ufunc;

pub use self::array::*;
pub use self::objects::*;
pub use self::types::*;
pub use self::ufunc::*;

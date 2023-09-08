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

use std::mem::forget;
use std::os::raw::c_void;

use pyo3::{
    types::{PyCapsule, PyModule},
    Py, PyResult, PyTryInto, Python,
};

fn get_numpy_api<'py>(
    py: Python<'py>,
    module: &str,
    capsule: &str,
) -> PyResult<*const *const c_void> {
    let module = PyModule::import(py, module)?;
    let capsule: &PyCapsule = PyTryInto::try_into(module.getattr(capsule)?)?;

    let api = capsule.pointer() as *const *const c_void;

    // Intentionally leak a reference to the capsule
    // so we can safely cache a pointer into its interior.
    forget(Py::<PyCapsule>::from(capsule));

    Ok(api)
}

// Implements wrappers for NumPy's Array and UFunc API
macro_rules! impl_api {
    [$offset: expr; $fname: ident ( $($arg: ident : $t: ty),* $(,)?) $( -> $ret: ty )* ] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $( -> $ret )* {
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

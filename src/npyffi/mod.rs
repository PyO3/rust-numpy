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
use std::os::raw::c_uint;
use std::os::raw::c_void;

use pyo3::{
    types::{PyAnyMethods, PyCapsule, PyCapsuleMethods, PyModule},
    PyResult, Python,
};

#[cfg(not(any(feature = "numpy-1", feature = "numpy-2")))]
compile_error!("at least one of feature \"numpy-1\" and feature \"numpy-2\" must be enabled");

pub const NPY_2_0_API_VERSION: c_uint = 0x00000012;

pub static ABI_API_VERSIONS: std::sync::OnceLock<(c_uint, c_uint)> = std::sync::OnceLock::new();

fn get_numpy_api<'py>(
    py: Python<'py>,
    module: &str,
    capsule: &str,
) -> PyResult<*const *const c_void> {
    let module = PyModule::import_bound(py, module)?;
    let capsule = module.getattr(capsule)?.downcast_into::<PyCapsule>()?;

    let api = capsule.pointer() as *const *const c_void;

    // Intentionally leak a reference to the capsule
    // so we can safely cache a pointer into its interior.
    forget(capsule);

    ABI_API_VERSIONS.get_or_init(|| {
        let abi_version = unsafe {
            #[allow(non_snake_case)]
            let PyArray_GetNDArrayCVersion = api.offset(0) as *const extern fn () -> c_uint;
            (*PyArray_GetNDArrayCVersion)()
        };
        let api_version = unsafe {
            #[allow(non_snake_case)]
            let PyArray_GetNDArrayCFeatureVersion = api.offset(211) as *const extern fn () -> c_uint;
            (*PyArray_GetNDArrayCFeatureVersion)()
        };
        #[cfg(all(feature = "numpy-1", not(feature = "numpy-2")))]
        if api_version >= NPY_2_0_API_VERSION {
            panic!(
                "the extension was compiled for numpy 1.x but the runtime version is 2.x (ABI {:08x}.{:08x})",
                abi_version,
                api_version
            );
        }
        #[cfg(all(not(feature = "numpy-1"), feature = "numpy-2"))]
        if api_version < NPY_2_0_API_VERSION {
            panic!(
                "the extension was compiled for numpy 2.x but the runtime version is 1.x (ABI {:08x}.{:08x})",
                abi_version,
                api_version
            );
        }
        (abi_version, api_version)
    });

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

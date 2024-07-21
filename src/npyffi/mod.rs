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
use std::os::raw::{c_uint, c_void};

use pyo3::{
    sync::GILOnceCell,
    types::{PyAnyMethods, PyCapsule, PyCapsuleMethods, PyModule},
    PyResult, Python,
};

pub const API_VERSION_2_0: c_uint = 0x00000012;

pub static API_VERSION: GILOnceCell<c_uint> = GILOnceCell::new();

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

    API_VERSION.get_or_init(py, || unsafe {
        #[allow(non_snake_case)]
        let PyArray_GetNDArrayCFeatureVersion = api.offset(211) as *const extern "C" fn() -> c_uint;
        (*PyArray_GetNDArrayCFeatureVersion)()
    });

    Ok(api)
}

const fn api_version_to_numpy_version_range(api_version: c_uint) -> (&'static str, &'static str) {
    match api_version {
        0..=0x00000008 => ("?", "1.7"),
        0x00000009 => ("1.8", "1.9"),
        0x0000000A => ("1.10", "1.12"),
        0x0000000B => ("1.13", "1.13"),
        0x0000000C => ("1.14", "1.15"),
        0x0000000D => ("1.16", "1.19"),
        0x0000000E => ("1.20", "1.21"),
        0x0000000F => ("1.22", "1.22"),
        0x00000010 => ("1.23", "1.24"),
        0x00000011 => ("1.25", "1.26"),
        0x00000012..=c_uint::MAX => ("2.0", "?"),
    }
}

// Implements wrappers for NumPy's Array and UFunc API
macro_rules! impl_api {
    // API available on all versions
    [$offset: expr; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            let fptr = self.get(py, $offset) as *const extern fn ($($arg : $t), *) $(-> $ret)*;
            (*fptr)($($arg), *)
        }
    };

    // API with version constraints, checked at runtime
    [$offset: expr; ..=1.26; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        impl_api![$offset; ..=0x00000011; $fname($($arg : $t), *) $(-> $ret)*];
    };
    [$offset: expr; 2.0..; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        impl_api![$offset; 0x00000012..; $fname($($arg : $t), *) $(-> $ret)*];
    };
    [$offset: expr; $($minimum: literal)?..=$($maximum: literal)?; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            let api_version = *API_VERSION.get(py).expect("API_VERSION is initialized");
            $(if api_version < $minimum { panic!(
                "{} requires API {:08X} or greater (NumPy {} or greater) but the runtime version is API {:08X}",
                stringify!($fname),
                $minimum,
                api_version_to_numpy_version_range($minimum).0,
                api_version,
            ) } )?
            $(if api_version > $maximum { panic!(
                "{} requires API {:08X} or lower (NumPy {} or lower) but the runtime version is API {:08X}",
                stringify!($fname),
                $maximum,
                api_version_to_numpy_version_range($maximum).1,
                api_version,
            ) } )?
            let fptr = self.get(py, $offset) as *const extern fn ($($arg: $t), *) $(-> $ret)*;
            (*fptr)($($arg), *)
        }
    };
    [$offset: expr; $($minimum: literal)?..; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        impl_api![$offset; $($minimum)?..=; $fname($($arg : $t), *) $(-> $ret)*];
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

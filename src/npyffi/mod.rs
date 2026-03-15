//! Low-Level bindings for NumPy C API.
//!
//! This module provides FFI bindings to [NumPy C API], implementing access to the NumPy array and
//! ufunc functionality. This binding is compatible with ABI v2 and the target API is v1.15 to
//! ensure the compatibility with the older NumPy version. See the official NumPy documentation
//! for more details about [API compatibility].
//!
//! [NumPy's C API]: https://numpy.org/doc/stable/reference/c-api
//! [API compatibility]: https://numpy.org/doc/stable/dev/depending_on_numpy.html
//!
#![allow(
    non_camel_case_types,
    missing_docs,
    missing_debug_implementations,
    clippy::too_many_arguments,
    clippy::missing_safety_doc
)]

use std::mem::forget;
use std::os::raw::{c_uint, c_void};
use std::ptr::NonNull;

use pyo3::{
    ffi::PyTypeObject,
    sync::PyOnceLock,
    types::{PyAnyMethods, PyCapsule, PyCapsuleMethods, PyModule},
    PyResult, Python,
};

static API_VERSION: PyOnceLock<c_uint> = PyOnceLock::new();

fn get_numpy_api<'py>(
    py: Python<'py>,
    module: &str,
    capsule: &str,
) -> PyResult<NonNull<*const c_void>> {
    let module = PyModule::import(py, module)?;
    let capsule = module.getattr(capsule)?.cast_into::<PyCapsule>()?;

    let api = capsule.pointer_checked(None)?;

    // Intentionally leak a reference to the capsule
    // so we can safely cache a pointer into its interior.
    forget(capsule);

    Ok(api.cast())
}

/// Returns whether the runtime `numpy` version is 2.0 or greater.
pub fn is_numpy_2<'py>(py: Python<'py>) -> bool {
    let api_version = *API_VERSION.get_or_init(py, || unsafe {
        PY_ARRAY_API.PyArray_GetNDArrayCFeatureVersion(py)
    });
    api_version >= NPY_2_0_API_VERSION
}

// Implements wrappers for NumPy's Array and UFunc API
macro_rules! impl_api {
    // API available on all versions
    [$offset: expr; $fname: ident ($($arg: ident: $t: ty),* $(,)?) $(-> $ret: ty)?] => {
        #[allow(non_snake_case)]
        pub unsafe fn $fname<'py>(&self, py: Python<'py>, $($arg : $t), *) $(-> $ret)* {
            let f: extern "C" fn ($($arg : $t), *) $(-> $ret)* = self.get(py, $offset).cast().read();
            f($($arg), *)
        }
    };
}

// Define type objects associated with the NumPy API
macro_rules! impl_array_type {
    ($(($api:ident [ $offset:expr ] , $tname:ident)),* $(,)?) => {
        /// All type objects exported by the NumPy API.
        #[allow(non_camel_case_types)]
        pub enum NpyTypes { $($tname),* }

        /// Get a pointer of the type object associated with `ty`.
        pub unsafe fn get_type_object<'py>(py: Python<'py>, ty: NpyTypes) -> *mut PyTypeObject {
            match ty {
                $( NpyTypes::$tname => $api.get(py, $offset).read() as _ ),*
            }
        }
    }
}

impl_array_type! {
    // Multiarray API
    // Slot 1 was never meaningfully used by NumPy
    (PY_ARRAY_API[2], PyArray_Type),
    (PY_ARRAY_API[3], PyArrayDescr_Type),
    // Unused slot 4, was `PyArrayFlags_Type`
    (PY_ARRAY_API[5], PyArrayIter_Type),
    (PY_ARRAY_API[6], PyArrayMultiIter_Type),
    // (PY_ARRAY_API[7], NPY_NUMUSERTYPES) -> c_int,
    (PY_ARRAY_API[8], PyBoolArrType_Type),
    // (PY_ARRAY_API[9], _PyArrayScalar_BoolValues) -> *mut PyBoolScalarObject,
    (PY_ARRAY_API[10], PyGenericArrType_Type),
    (PY_ARRAY_API[11], PyNumberArrType_Type),
    (PY_ARRAY_API[12], PyIntegerArrType_Type),
    (PY_ARRAY_API[13], PySignedIntegerArrType_Type),
    (PY_ARRAY_API[14], PyUnsignedIntegerArrType_Type),
    (PY_ARRAY_API[15], PyInexactArrType_Type),
    (PY_ARRAY_API[16], PyFloatingArrType_Type),
    (PY_ARRAY_API[17], PyComplexFloatingArrType_Type),
    (PY_ARRAY_API[18], PyFlexibleArrType_Type),
    (PY_ARRAY_API[19], PyCharacterArrType_Type),
    (PY_ARRAY_API[20], PyByteArrType_Type),
    (PY_ARRAY_API[21], PyShortArrType_Type),
    (PY_ARRAY_API[22], PyIntArrType_Type),
    (PY_ARRAY_API[23], PyLongArrType_Type),
    (PY_ARRAY_API[24], PyLongLongArrType_Type),
    (PY_ARRAY_API[25], PyUByteArrType_Type),
    (PY_ARRAY_API[26], PyUShortArrType_Type),
    (PY_ARRAY_API[27], PyUIntArrType_Type),
    (PY_ARRAY_API[28], PyULongArrType_Type),
    (PY_ARRAY_API[29], PyULongLongArrType_Type),
    (PY_ARRAY_API[30], PyFloatArrType_Type),
    (PY_ARRAY_API[31], PyDoubleArrType_Type),
    (PY_ARRAY_API[32], PyLongDoubleArrType_Type),
    (PY_ARRAY_API[33], PyCFloatArrType_Type),
    (PY_ARRAY_API[34], PyCDoubleArrType_Type),
    (PY_ARRAY_API[35], PyCLongDoubleArrType_Type),
    (PY_ARRAY_API[36], PyObjectArrType_Type),
    (PY_ARRAY_API[37], PyStringArrType_Type),
    (PY_ARRAY_API[38], PyUnicodeArrType_Type),
    (PY_ARRAY_API[39], PyVoidArrType_Type),
    (PY_ARRAY_API[214], PyTimeIntegerArrType_Type),
    (PY_ARRAY_API[215], PyDatetimeArrType_Type),
    (PY_ARRAY_API[216], PyTimedeltaArrType_Type),
    (PY_ARRAY_API[217], PyHalfArrType_Type),
    (PY_ARRAY_API[218], NpyIter_Type),
    // UFunc API
    (PY_UFUNC_API[0], PyUFunc_Type),
}

pub mod array;
pub mod flags;
mod npy_common;
mod numpyconfig;
pub mod objects;
pub mod types;
pub mod ufunc;

pub use self::array::*;
pub use self::flags::*;
pub use self::npy_common::*;
pub use self::numpyconfig::*;
pub use self::objects::*;
pub use self::types::*;
pub use self::ufunc::*;

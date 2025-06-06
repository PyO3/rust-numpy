//! Safe interface for NumPy's random [`BitGenerator`][]
//!
//! `BitGenerator`: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html

use pyo3::{ffi, prelude::*, sync::GILOnceCell, types::PyType, PyTypeInfo};

use crate::npyffi::get_bitgen_api;

///! Wrapper for NumPy's random [`BitGenerator`][]
/// 
///! [BitGenerator]: https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.BitGenerator.html
#[repr(transparent)]
pub struct BitGenerator(PyAny);

unsafe impl PyTypeInfo for BitGenerator {
    const NAME: &'static str = "BitGenerator";
    const MODULE: Option<&'static str> = Some("numpy.random");

    fn type_object_raw<'py>(py: Python<'py>) -> *mut ffi::PyTypeObject {
        const CLS: GILOnceCell<Py<PyType>> = GILOnceCell::new();
        let cls = CLS
            .get_or_try_init::<_, PyErr>(py, || {
                Ok(py
                    .import("numpy.random")?
                    .getattr("BitGenerator")?
                    .downcast_into::<PyType>()?
                    .unbind())
            })
            .expect("Failed to get BitGenerator type object")
            .clone_ref(py)
            .into_bound(py);
        cls.as_type_ptr()
    }
}


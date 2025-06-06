//! Safe interface for NumPy's random [`BitGenerator`]

use pyo3::{ffi, prelude::*, sync::GILOnceCell, types::PyType, PyTypeInfo};

use crate::npyffi::get_bitgen_api;

///! Wrapper for NumPy's random [`BitGenerator`][bg]
///!
///! [bg]: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html
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

/// Methods for [`BitGenerator`]
pub trait BitGeneratorMethods {
    /// Returns the next random unsigned 64 bit integer
    fn next_uint64(&self) -> u64;
    /// Returns the next random unsigned 32 bit integer
    fn next_uint32(&self) -> u32;
    /// Returns the next random double
    fn next_double(&self) -> libc::c_double;
    /// Returns the next raw value (can be used for testing)
    fn next_raw(&self) -> u64;
}

// TODO: cache npy_bitgen pointer
impl<'py> BitGeneratorMethods for Bound<'py, BitGenerator> {
    fn next_uint64(&self) -> u64 {
        todo!()
    }
    fn next_uint32(&self) -> u32 {
        todo!()
    }
    fn next_double(&self) -> libc::c_double {
        todo!()
    }
    fn next_raw(&self) -> u64 {
        let mut api = get_bitgen_api(self.as_any()).expect("Could not get bitgen");
        unsafe {
            let api = api.as_mut();
            (api.next_raw)(api.state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitgen() -> PyResult<()> {
        Python::with_gil(|py| {
            let default_rng = py.import("numpy.random")?.getattr("default_rng")?;
            let bitgen = default_rng.call0()?.getattr("bit_generator")?.downcast_into::<BitGenerator>()?;
            let res = bitgen.next_raw();
            dbg!(res);
            Ok(())
        })
    }   
}

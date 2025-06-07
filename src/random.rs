//! Safe interface for NumPy's random [`BitGenerator`]

use pyo3::{ffi, prelude::*, sync::GILOnceCell, types::{PyCapsule, PyType}, PyTypeInfo, exceptions::PyRuntimeError};

use crate::npyffi::npy_bitgen;

///! Wrapper for [`np.random.BitGenerator`][bg]
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
pub trait BitGeneratorMethods<'py> {
    /// Returns a new [`BitGen`]
    fn bit_gen(&self) -> PyResult<BitGen<'py>>;
}

impl<'py> BitGeneratorMethods<'py> for Bound<'py, BitGenerator> {
    fn bit_gen(&self) -> PyResult<BitGen<'py>> {
        let capsule = self.as_any().getattr("capsule")?.downcast_into::<PyCapsule>()?;
        assert_eq!(capsule.name()?, Some(c"BitGenerator"));
        let ptr = capsule.pointer() as *mut npy_bitgen;
        // SAFETY: the lifetime of `ptr` is derived from the lifetime of `self`
        let ref_ = unsafe { ptr.as_mut::<'py>() }.ok_or_else(|| PyRuntimeError::new_err("Invalid BitGenerator capsule"))?;
        Ok(BitGen(ref_))
    }
}

/// Wrapper for [`npy_bitgen`]
pub struct BitGen<'a>(&'a mut npy_bitgen);

impl<'py> BitGen<'py> {
    /// Returns the next random unsigned 64 bit integer
    pub fn next_uint64(&self) -> u64 {
        unsafe { (self.0.next_uint64)(self.0.state) }
    }
    /// Returns the next random unsigned 32 bit integer
    pub fn next_uint32(&self) -> u32 {
        unsafe { (self.0.next_uint32)(self.0.state) }
    }
    /// Returns the next random double
    pub fn next_double(&self) -> libc::c_double {
        unsafe { (self.0.next_double)(self.0.state) }
    }
    /// Returns the next raw value (can be used for testing)
    pub fn next_raw(&self) -> u64 {
        unsafe { (self.0.next_raw)(self.0.state) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitgen() -> PyResult<()> {
        Python::with_gil(|py| {
            let default_rng = py.import("numpy.random")?.getattr("default_rng")?;
            let bitgen = default_rng.call0()?.getattr("bit_generator")?.downcast_into::<BitGenerator>()?.bit_gen()?;
            let res = bitgen.next_raw();
            dbg!(res);
            Ok(())
        })
    }   
}

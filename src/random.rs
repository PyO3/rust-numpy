//! Safe interface for NumPy's random [`BitGenerator`][bg].
//!
//! Using the patterns described in [“Extending `numpy.random`”][ext],
//! you can generate random numbers without holding the GIL,
//! by [acquiring][`PyBitGeneratorMethods::lock`] a [lock][`PyBitGeneratorLock`] for the [`PyBitGenerator`]:
//!
//! ```rust
//! use pyo3::prelude::*;
//! use numpy::random::{PyBitGenerator, PyBitGeneratorMethods as _};
//!
//! let mut bitgen = Python::with_gil(|py| -> PyResult<_> {
//!     let default_rng = py.import("numpy.random")?.getattr("default_rng")?.call0()?;
//!     let bit_generator = default_rng.getattr("bit_generator")?.downcast_into::<PyBitGenerator>()?;
//!     bit_generator.lock()
//! })?;
//! let random_number = bitgen.next_u64();
//! ```
//!
//! With the [`rand`] crate installed, you can also use the [`rand::Rng`] APIs from the [`PyBitGeneratorLock`]:
//!
//! ```rust
//! use rand::Rng as _;
//!
//! if bitgen.random_ratio(1, 1_000_000) {
//!     println!("a sure thing");
//! }
//! ```
//!
//! [bg]: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html
//! [ext]: https://numpy.org/doc/stable/reference/random/extending.html

use std::ptr::NonNull;

use pyo3::{
    exceptions::PyRuntimeError,
    ffi,
    prelude::*,
    sync::GILOnceCell,
    types::{DerefToPyAny, PyCapsule, PyType},
    PyTypeInfo,
};

use crate::npyffi::npy_bitgen;

/// Wrapper for [`np.random.BitGenerator`][bg].
///
/// [bg]: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html
#[repr(transparent)]
pub struct PyBitGenerator(PyAny);

impl DerefToPyAny for PyBitGenerator {}

unsafe impl PyTypeInfo for PyBitGenerator {
    const NAME: &'static str = "PyBitGenerator";
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

/// Methods for [`PyBitGenerator`].
pub trait PyBitGeneratorMethods {
    /// Acquire a lock on the BitGenerator to allow calling its methods in.
    fn lock(&self) -> PyResult<PyBitGeneratorLock>;
}

impl<'py> PyBitGeneratorMethods for Bound<'py, PyBitGenerator> {
    fn lock(&self) -> PyResult<PyBitGeneratorLock> {
        let capsule = self.getattr("capsule")?.downcast_into::<PyCapsule>()?;
        let lock = self.getattr("lock")?;
        if lock.getattr("locked")?.call0()?.extract()? {
            return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
        }
        lock.getattr("acquire")?.call0()?;

        assert_eq!(capsule.name()?, Some(c"BitGenerator"));
        let ptr = capsule.pointer() as *mut npy_bitgen;
        let non_null = match NonNull::new(ptr) {
            Some(non_null) => non_null,
            None => {
                lock.getattr("release")?.call0()?;
                return Err(PyRuntimeError::new_err("Invalid BitGenerator capsule"));
            }
        };
        Ok(PyBitGeneratorLock {
            raw_bitgen: non_null,
            lock: lock.unbind(),
        })
    }
}

impl<'py> TryFrom<&Bound<'py, PyBitGenerator>> for PyBitGeneratorLock {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyBitGenerator>) -> Result<Self, Self::Error> {
        value.lock()
    }
}

/// [`PyBitGenerator`] lock allowing to access its methods without holding the GIL.
pub struct PyBitGeneratorLock {
    raw_bitgen: NonNull<npy_bitgen>,
    lock: Py<PyAny>,
}

// SAFETY for all methods: We hold the BitGenerator lock, so nothing is allowed to change its state
impl PyBitGeneratorLock {
    /// Returns the next random unsigned 64 bit integer.
    pub fn next_uint64(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw_bitgen.as_mut();
            (bitgen.next_uint64)(bitgen.state)
        }
    }
    /// Returns the next random unsigned 32 bit integer.
    pub fn next_uint32(&mut self) -> u32 {
        unsafe {
            let bitgen = self.raw_bitgen.as_mut();
            (bitgen.next_uint32)(bitgen.state)
        }
    }
    /// Returns the next random double.
    pub fn next_double(&mut self) -> libc::c_double {
        unsafe {
            let bitgen = self.raw_bitgen.as_mut();
            (bitgen.next_double)(bitgen.state)
        }
    }
    /// Returns the next raw value (can be used for testing).
    pub fn next_raw(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw_bitgen.as_mut();
            (bitgen.next_raw)(bitgen.state)
        }
    }
}

impl Drop for PyBitGeneratorLock {
    fn drop(&mut self) {
        let r = Python::with_gil(|py| -> PyResult<()> {
            self.lock.bind(py).getattr("release")?.call0()?;
            Ok(())
        });
        if let Err(e) = r {
            eprintln!("Failed to release BitGenerator lock: {e}");
        }
    }
}

#[cfg(feature = "rand")]
impl rand::RngCore for PyBitGeneratorLock {
    fn next_u32(&mut self) -> u32 {
        self.next_uint32()
    }
    fn next_u64(&mut self) -> u64 {
        self.next_uint64()
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        rand::rand_core::impls::fill_bytes_via_next(self, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_bit_generator<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
        let default_rng = py.import("numpy.random")?.getattr("default_rng")?.call0()?;
        let bit_generator = default_rng
            .getattr("bit_generator")?
            .downcast_into::<PyBitGenerator>()?;
        Ok(bit_generator)
    }

    #[test]
    fn bitgen() -> PyResult<()> {
        let mut bitgen = Python::with_gil(|py| get_bit_generator(py)?.lock())?;
        let _ = bitgen.next_raw();
        std::mem::drop(bitgen);
        Ok(())
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        let mut bitgen = Python::with_gil(|py| get_bit_generator(py)?.lock())?;
        assert!(bitgen.random_ratio(1, 1));
        assert!(!bitgen.random_ratio(0, 1));
        std::mem::drop(bitgen);
        Ok(())
    }

    /// Test that dropping the lock works while holding the GIL
    #[test]
    fn unlock_with_held_gil() -> PyResult<()> {
        Python::with_gil(|py| {
            let generator = get_bit_generator(py)?;
            let mut bitgen = generator.lock()?;
            let _ = bitgen.next_raw();
            std::mem::drop(bitgen);
            Ok(())
        })
    }

    #[test]
    fn double_lock_fails() -> PyResult<()> {
        Python::with_gil(|py| {
            let generator = get_bit_generator(py)?;
            let d1 = generator.lock()?;
            assert!(generator.lock().is_err());
            std::mem::drop(d1);
            Ok(())
        })
    }
}

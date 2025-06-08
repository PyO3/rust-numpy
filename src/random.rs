//! Safe interface for NumPy's random [`BitGenerator`][bg].
//!
//! Using the patterns described in [“Extending `numpy.random`”][ext],
//! you can generate random numbers without holding the GIL,
//! by [acquiring][`PyBitGeneratorMethods::lock`] a lock [guard][`PyBitGeneratorGuard`] for the [`PyBitGenerator`]:
//!
//! ```
//! use pyo3::prelude::*;
//! use numpy::random::{PyBitGenerator, PyBitGeneratorMethods as _};
//!
//! fn default_bit_gen<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
//!     let default_rng = py.import("numpy.random")?.call_method0("default_rng")?;
//!     let bit_generator = default_rng.getattr("bit_generator")?.downcast_into()?;
//!     Ok(bit_generator)
//! }
//!
//! let random_number = Python::with_gil(|py| -> PyResult<_> {
//!     let mut bitgen = default_bit_gen(py)?.lock()?;
//!     // use bitgen without holding the GIL
//!     Ok(py.allow_threads(|| bitgen.next_uint64()))
//! })?;
//! # Ok::<(), PyErr>(())
//! ```
//!
//! With the [`rand`] crate installed, you can also use the [`rand::Rng`] APIs from the [`PyBitGeneratorGuard`]:
//!
//! ```
//! # use pyo3::prelude::*;
//! use rand::Rng as _;
//! # use numpy::random::{PyBitGenerator, PyBitGeneratorMethods as _};
//! # // TODO: reuse function definition from above?
//! # fn default_bit_gen<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
//! #     let default_rng = py.import("numpy.random")?.call_method0("default_rng")?;
//! #     let bit_generator = default_rng.getattr("bit_generator")?.downcast_into()?;
//! #     Ok(bit_generator)
//! # }
//!
//! Python::with_gil(|py| -> PyResult<_> {
//!     let mut bitgen = default_bit_gen(py)?.lock()?;
//!     if bitgen.random_ratio(1, 1_000_000) {
//!         println!("a sure thing");
//!     }
//!     Ok(())
//! })?;
//! # Ok::<(), PyErr>(())
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
/// See also [`PyBitGeneratorMethods`].
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
pub trait PyBitGeneratorMethods<'py> {
    /// Acquire a lock on the BitGenerator to allow calling its methods in.
    fn lock(&self) -> PyResult<PyBitGeneratorGuard<'py>>;
}

impl<'py> PyBitGeneratorMethods<'py> for Bound<'py, PyBitGenerator> {
    fn lock(&self) -> PyResult<PyBitGeneratorGuard<'py>> {
        let capsule = self.getattr("capsule")?.downcast_into::<PyCapsule>()?;
        let lock = self.getattr("lock")?;
        if lock.call_method0("locked")?.extract()? {
            return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
        }
        lock.call_method0("acquire")?;

        assert_eq!(capsule.name()?, Some(c"BitGenerator"));
        let ptr = capsule.pointer() as *mut npy_bitgen;
        let non_null = match NonNull::new(ptr) {
            Some(non_null) => non_null,
            None => {
                lock.call_method0("release")?;
                return Err(PyRuntimeError::new_err("Invalid BitGenerator capsule"));
            }
        };
        Ok(PyBitGeneratorGuard {
            raw_bitgen: non_null,
            _capsule: capsule.unbind(),
            lock: lock.unbind(),
            py: self.py(),
        })
    }
}

impl<'py> TryFrom<&Bound<'py, PyBitGenerator>> for PyBitGeneratorGuard<'py> {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyBitGenerator>) -> Result<Self, Self::Error> {
        value.lock()
    }
}

/// [`PyBitGenerator`] lock allowing to access its methods without holding the GIL.
pub struct PyBitGeneratorGuard<'py> {
    raw_bitgen: NonNull<npy_bitgen>,
    /// This field makes sure the `raw_bitgen` inside the capsule doesn’t get deallocated.
    _capsule: Py<PyCapsule>,
    /// This lock makes sure no other threads try to use the BitGenerator while we do.
    lock: Py<PyAny>,
    /// This should be an unsafe field (https://github.com/rust-lang/rust/issues/132922)
    ///
    /// SAFETY: only use this in `Drop::drop` (when we are sure the GIL is held).
    py: Python<'py>,
}

// SAFETY: we can’t have public APIs that access the Python objects,
// only the `raw_bitgen` pointer.
unsafe impl Send for PyBitGeneratorGuard<'_> {}

impl Drop for PyBitGeneratorGuard<'_> {
    fn drop(&mut self) {
        // ignore errors. This includes when `try_drop` was called manually
        let _ = self.lock.bind(self.py).call_method0("release");
    }
}

// SAFETY: We hold the `BitGenerator.lock`,
// so nothing apart from us is allowed to change its state.
impl<'py> PyBitGeneratorGuard<'py> {
    /// Drop the lock manually before `Drop::drop` tries to do it (used for testing).
    #[allow(dead_code)]
    fn try_drop(self, py: Python<'py>) -> PyResult<()> {
        self.lock.bind(py).call_method0("release")?;
        Ok(())
    }

    /// Returns the next random unsigned 64 bit integer.
    pub fn next_uint64(&mut self) -> u64 {
        unsafe {
            let bitgen = *self.raw_bitgen.as_ptr();
            (bitgen.next_uint64)(bitgen.state)
        }
    }
    /// Returns the next random unsigned 32 bit integer.
    pub fn next_uint32(&mut self) -> u32 {
        unsafe {
            let bitgen = *self.raw_bitgen.as_ptr();
            (bitgen.next_uint32)(bitgen.state)
        }
    }
    /// Returns the next random double.
    pub fn next_double(&mut self) -> libc::c_double {
        unsafe {
            let bitgen = *self.raw_bitgen.as_ptr();
            (bitgen.next_double)(bitgen.state)
        }
    }
    /// Returns the next raw value (can be used for testing).
    pub fn next_raw(&mut self) -> u64 {
        unsafe {
            let bitgen = *self.raw_bitgen.as_ptr();
            (bitgen.next_raw)(bitgen.state)
        }
    }
}

#[cfg(feature = "rand")]
impl rand::RngCore for PyBitGeneratorGuard<'_> {
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
        let default_rng = py.import("numpy.random")?.call_method0("default_rng")?;
        let bit_generator = default_rng
            .getattr("bit_generator")?
            .downcast_into::<PyBitGenerator>()?;
        Ok(bit_generator)
    }

    /// Test the primary use case: acquire the lock, release the GIL, then use the lock
    #[test]
    fn use_outside_gil() -> PyResult<()> {
        Python::with_gil(|py| {
            let mut bitgen = get_bit_generator(py)?.lock()?;
            py.allow_threads(|| {
                let _ = bitgen.next_raw();
            });
            assert!(bitgen.try_drop(py).is_ok());
            Ok(())
        })
    }

    /// More complex version of primary use case: use from multiple threads
    #[cfg(feature = "rand")]
    #[test]
    fn use_parallel() -> PyResult<()> {
        use crate::array::{PyArray2, PyArrayMethods as _};
        use ndarray::Dimension;
        use rand::Rng;
        use std::sync::{Arc, Mutex};

        Python::with_gil(|py| -> PyResult<_> {
            let mut arr = PyArray2::<u32>::zeros(py, (2, 300), false).readwrite();
            let bitgen = get_bit_generator(py)?.lock()?;
            let bitgen = Arc::new(Mutex::new(bitgen));

            let (_n_threads, chunk_size) = arr.dims().into_pattern();
            let slice = arr.as_slice_mut()?;

            Python::allow_threads(py, || {
                std::thread::scope(|s| {
                    for chunk in slice.chunks_exact_mut(chunk_size) {
                        let bitgen = Arc::clone(&bitgen);
                        s.spawn(move || {
                            let mut bitgen = bitgen.lock().unwrap();
                            chunk.fill_with(|| bitgen.random_range(10..200));
                        });
                    }
                })
            });
            Ok(())
        })
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        Python::with_gil(|py| {
            let mut bitgen = get_bit_generator(py)?.lock()?;
            py.allow_threads(|| {
                assert!(bitgen.random_ratio(1, 1));
                assert!(!bitgen.random_ratio(0, 1));
            });
            assert!(bitgen.try_drop(py).is_ok());
            Ok(())
        })
    }

    #[test]
    fn double_lock_fails() -> PyResult<()> {
        Python::with_gil(|py| {
            let generator = get_bit_generator(py)?;
            let bitgen = generator.lock()?;
            assert!(generator.lock().is_err());
            assert!(bitgen.try_drop(py).is_ok());
            Ok(())
        })
    }
}

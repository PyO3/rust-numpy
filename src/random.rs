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
//!     let bit_generator = default_rng.getattr("bit_generator")?.cast_into()?;
//!     Ok(bit_generator)
//! }
//!
//! let random_number = Python::attach(|py| -> PyResult<_> {
//!     let mut bitgen = default_bit_gen(py)?.lock()?;
//!     // use bitgen without holding the GIL
//!     let r = py.detach(|| bitgen.next_u64());
//!     // release the lock manually while holding the GIL again
//!     bitgen.release(py)?;
//!     Ok(r)
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
//! #     let bit_generator = default_rng.getattr("bit_generator")?.cast_into()?;
//! #     Ok(bit_generator)
//! # }
//!
//! Python::attach(|py| -> PyResult<_> {
//!     let mut bitgen = default_bit_gen(py)?.lock()?;
//!     if bitgen.random_ratio(1, 1_000_000) {
//!         println!("a sure thing");
//!     }
//!     bitgen.release(py)?;
//!     Ok(())
//! })?;
//! # Ok::<(), PyErr>(())
//! ```
//!
//! [bg]: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html
//! [ext]: https://numpy.org/doc/stable/reference/random/extending.html

use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use pyo3::{
    exceptions::PyRuntimeError,
    ffi, intern,
    prelude::*,
    sync::PyOnceLock,
    types::{DerefToPyAny, PyCapsule, PyType},
    PyTypeInfo,
};

use crate::npyffi::bitgen_t;

/// Addresses of the `bitgen_t`s that currently have a live [`PyBitGeneratorGuard`].
fn locked_bitgens() -> &'static Mutex<HashSet<usize>> {
    static LOCKED: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    LOCKED.get_or_init(|| Mutex::new(HashSet::new()))
}

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
        static CLS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        let cls = CLS
            .import(py, "numpy.random", "BitGenerator")
            .expect("Failed to get BitGenerator type object");
        cls.as_type_ptr()
    }
}

/// Methods for [`PyBitGenerator`].
pub trait PyBitGeneratorMethods {
    /// Acquire a lock on the BitGenerator to allow calling its methods in.
    fn lock(&self) -> PyResult<PyBitGeneratorGuard>;
}

impl<'py> PyBitGeneratorMethods for Bound<'py, PyBitGenerator> {
    fn lock(&self) -> PyResult<PyBitGeneratorGuard> {
        let py = self.py();
        let capsule = self
            .getattr(intern!(py, "capsule"))?
            .cast_into::<PyCapsule>()?;
        let lock = self.getattr(intern!(py, "lock"))?;
        // Acquire the (reentrant!) lock in non-blocking mode or return an error.
        // We also check `locked_bitgens` later to prevent same-thread re-locking.
        if !lock
            .call_method(intern!(py, "acquire"), (false,), None)?
            .extract()?
        {
            return Err(PyRuntimeError::new_err(
                "Failed to acquire BitGenerator lock",
            ));
        }
        // Return the guard or release the lock if the capsule is invalid
        let raw_bitgen = match capsule.pointer_checked(Some(ffi::c_str!("BitGenerator"))) {
            Ok(ptr) => ptr.cast(),
            Err(_) => {
                lock.call_method0(intern!(py, "release"))?;
                return Err(PyRuntimeError::new_err("Invalid BitGenerator capsule"));
            }
        };
        // Reject re-locking the same `BitGenerator`, since the `RLock` above won’t.
        if !locked_bitgens()
            .lock()
            .unwrap()
            .insert(raw_bitgen.as_ptr() as usize)
        {
            lock.call_method0(intern!(py, "release"))?;
            return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
        }
        Ok(PyBitGeneratorGuard {
            raw_bitgen,
            released: false,
            _bit_generator: self.clone().unbind(),
            lock: lock.unbind(),
        })
    }
}

impl<'py> TryFrom<&Bound<'py, PyBitGenerator>> for PyBitGeneratorGuard {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyBitGenerator>) -> Result<Self, Self::Error> {
        value.lock()
    }
}

/// [`PyBitGenerator`] lock allowing to access its methods without holding the GIL.
///
/// Since [dropping](`Drop::drop`) this acquires the GIL,
/// prefer to call [`release`][`PyBitGeneratorGuard::release`] manually to release the lock.
pub struct PyBitGeneratorGuard {
    raw_bitgen: NonNull<bitgen_t>,
    /// Whether this guard has been manually released.
    released: bool,
    /// This field makes sure `raw_bitgen` doesn’t get deallocated: the capsule wraps a raw
    /// pointer into memory owned by the `BitGenerator` itself, with no back-reference of its
    /// own, so only keeping the parent object alive keeps that memory valid.
    _bit_generator: Py<PyBitGenerator>,
    /// This lock makes sure no other threads try to use the BitGenerator while we do.
    lock: Py<PyAny>,
}

// SAFETY: 1. We don’t hold the GIL, so we can’t access the Python objects.
//         2. We only access `raw_bitgen` from `&mut self`, which protects it from parallel access.
unsafe impl Send for PyBitGeneratorGuard {}

impl Drop for PyBitGeneratorGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        locked_bitgens()
            .lock()
            .unwrap()
            .remove(&(self.raw_bitgen.as_ptr() as usize));
        // ignore errors because `drop` can’t fail
        let _ = Python::attach(|py| -> PyResult<_> {
            self.lock.bind(py).call_method0(intern!(py, "release"))?;
            Ok(())
        });
    }
}

// SAFETY: 1. We hold the `BitGenerator.lock` and are the sole entry in `locked_bitgens` for
//            this `raw_bitgen`, so nothing apart from us is allowed to change its state.
//         2. We hold the `BitGenerator` itself, so its embedded state can’t be deallocated.
impl<'py> PyBitGeneratorGuard {
    /// Release the lock, allowing for checking for errors.
    pub fn release(mut self, py: Python<'py>) -> PyResult<()> {
        self.released = true; // only ever read by drop at the end of a scope (like this one).
        locked_bitgens()
            .lock()
            .unwrap()
            .remove(&(self.raw_bitgen.as_ptr() as usize));
        self.lock.bind(py).call_method0(intern!(py, "release"))?;
        Ok(())
    }

    /// Returns the next random unsigned 64 bit integer.
    pub fn next_u64(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw_bitgen.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_uint64)((*bitgen).state)
        }
    }
    /// Returns the next random unsigned 32 bit integer.
    pub fn next_u32(&mut self) -> u32 {
        unsafe {
            let bitgen = self.raw_bitgen.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_uint32)((*bitgen).state)
        }
    }
    /// Returns the next random double.
    pub fn next_double(&mut self) -> f64 {
        unsafe {
            let bitgen = self.raw_bitgen.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_double)((*bitgen).state)
        }
    }
    /// Returns the next raw value (can be used for testing).
    pub fn next_raw(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw_bitgen.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_raw)((*bitgen).state)
        }
    }
}

#[cfg(feature = "rand_core")]
impl rand_core::RngCore for PyBitGeneratorGuard {
    fn next_u32(&mut self) -> u32 {
        PyBitGeneratorGuard::next_u32(self)
    }
    fn next_u64(&mut self) -> u64 {
        PyBitGeneratorGuard::next_u64(self)
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_bit_generator<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
        let default_rng = py.import("numpy.random")?.call_method0("default_rng")?;
        let bit_generator = default_rng
            .getattr("bit_generator")?
            .cast_into::<PyBitGenerator>()?;
        Ok(bit_generator)
    }

    /// Test the primary use case: acquire the lock, release the GIL, then use the lock
    #[test]
    fn use_outside_gil() -> PyResult<()> {
        Python::attach(|py| {
            let mut bitgen = get_bit_generator(py)?.lock()?;
            py.detach(|| {
                let _ = bitgen.next_raw();
            });
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }

    /// More complex version of primary use case: use from multiple threads
    #[cfg(feature = "rand_core")]
    #[test]
    fn use_parallel() -> PyResult<()> {
        use crate::array::{PyArray2, PyArrayMethods as _};
        use ndarray::Dimension;
        use rand::Rng;
        use std::sync::{Arc, Mutex};

        Python::attach(|py| -> PyResult<_> {
            let mut arr = PyArray2::<u32>::zeros(py, (2, 300), false).readwrite();
            let bitgen = get_bit_generator(py)?.lock()?;
            let bitgen = Arc::new(Mutex::new(bitgen));

            let (_n_threads, chunk_size) = arr.dims().into_pattern();
            let slice = arr.as_slice_mut()?;

            py.detach(|| {
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

            Arc::into_inner(bitgen)
                .unwrap()
                .into_inner()
                .unwrap()
                .release(py)?;
            Ok(())
        })
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand_core")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        Python::attach(|py| {
            let mut bitgen = get_bit_generator(py)?.lock()?;
            py.detach(|| {
                assert!(bitgen.random_ratio(1, 1));
                assert!(!bitgen.random_ratio(0, 1));
            });
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }

    #[test]
    fn lock_keeps_bit_generator_alive() -> PyResult<()> {
        Python::attach(|py| {
            let generator = get_bit_generator(py)?;
            let sys = py.import("sys")?;
            let refcount_before: usize =
                sys.call_method1("getrefcount", (&generator,))?.extract()?;

            let bitgen = generator.lock()?;
            let refcount_locked: usize =
                sys.call_method1("getrefcount", (&generator,))?.extract()?;
            assert!(refcount_locked > refcount_before);

            bitgen.release(py)?;
            let refcount_after: usize =
                sys.call_method1("getrefcount", (&generator,))?.extract()?;
            assert_eq!(refcount_after, refcount_before);
            Ok(())
        })
    }

    #[test]
    fn double_lock_fails() -> PyResult<()> {
        Python::attach(|py| {
            let generator = get_bit_generator(py)?;
            let bitgen = generator.lock()?;
            assert!(generator.lock().is_err());
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }

    #[test]
    fn double_lock_fails_2() -> PyResult<()> {
        Python::attach(|py| {
            let generator1 = get_bit_generator(py)?;
            let generator2 = generator1.clone();
            assert_eq!(
                generator1
                    .getattr("capsule")?
                    .cast::<PyCapsule>()?
                    .pointer_checked(Some(ffi::c_str!("BitGenerator")))?,
                generator2
                    .getattr("capsule")?
                    .cast::<PyCapsule>()?
                    .pointer_checked(Some(ffi::c_str!("BitGenerator")))?
            );
            let bitgen = generator1.lock()?;
            assert!(generator2.lock().is_err());
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }
}

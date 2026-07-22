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
//! With the `rand` crate installed, you can also use its `Rng` APIs on any generator, since
//! [`BitGenerator`] implements [`rand_core::RngCore`] (and [`PyBitGeneratorGuard`] derefs to it).
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
use std::ops::{Deref, DerefMut};
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

    /// Spawn `n_children` independent child `BitGenerator`s.
    ///
    /// This is the recommended way to obtain generators for multiple threads: unlike sharing a
    /// single [locked][PyBitGeneratorMethods::lock] one, each child has its own, independent state.
    fn spawn(&self, n_children: usize) -> PyResult<Vec<BitGenerator>>;
}

impl<'py> PyBitGeneratorMethods for Bound<'py, PyBitGenerator> {
    fn lock(&self) -> PyResult<PyBitGeneratorGuard> {
        let py = self.py();
        let lock = self.getattr(intern!(py, "lock"))?;
        // Acquire the (reentrant!) lock in non-blocking mode or return an error.
        if !lock
            .call_method(intern!(py, "acquire"), (false,), None)?
            .extract()?
        {
            return Err(PyRuntimeError::new_err(
                "Failed to acquire BitGenerator lock",
            ));
        }
        // SAFETY: we hold the lock, and the guard keeps holding it for the `BitGenerator`’s
        //         lifetime; `locked_bitgens` below additionally rejects same-thread re-locking.
        let generator = match unsafe { BitGenerator::new(self.clone()) } {
            Ok(generator) => generator,
            Err(err) => {
                lock.call_method0(intern!(py, "release"))?;
                return Err(err);
            }
        };
        // Reject re-locking the same `BitGenerator`, since the `RLock` above won’t.
        if !locked_bitgens().lock().unwrap().insert(generator.addr()) {
            lock.call_method0(intern!(py, "release"))?;
            return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
        }
        Ok(PyBitGeneratorGuard {
            generator,
            released: false,
            lock: lock.unbind(),
        })
    }

    fn spawn(&self, n_children: usize) -> PyResult<Vec<BitGenerator>> {
        let py = self.py();
        self.call_method1(intern!(py, "spawn"), (n_children,))?
            .try_iter()?
            // SAFETY: each child is freshly spawned and only handed to us, so it’s exclusively ours.
            .map(|child| unsafe { BitGenerator::new(child?.cast_into::<PyBitGenerator>()?) })
            .collect()
    }
}

impl<'py> TryFrom<&Bound<'py, PyBitGenerator>> for PyBitGeneratorGuard {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyBitGenerator>) -> Result<Self, Self::Error> {
        value.lock()
    }
}

/// A numpy `BitGenerator` usable without the GIL, with exclusive access to its state.
///
/// [`spawn`][PyBitGeneratorMethods::spawn] hands out independent, owned ones; a
/// [`PyBitGeneratorGuard`] derefs to one borrowing a shared generator under its held lock.
/// [`share`][BitGenerator::share] locks an owned one so it can be used from Python again.
pub struct BitGenerator {
    raw: NonNull<bitgen_t>,
    /// Keeps `raw` alive: the capsule’s pointer lives in memory owned by the `BitGenerator`, which
    /// has no back-reference of its own, so only keeping it alive keeps that memory valid.
    _bit_generator: Py<PyBitGenerator>,
}

// SAFETY: `raw` is only ever accessed through `&mut self`, so it can’t be used in parallel, and we
//         keep its `bitgen_t` alive via `_bit_generator`. Every `BitGenerator` has exclusive access
//         to its `bitgen_t` (a fresh `spawn` child it owns, or a shared one protected by a held lock
//         inside a `PyBitGeneratorGuard`), so nothing else can touch its state.
unsafe impl Send for BitGenerator {}

impl BitGenerator {
    /// Extracts the raw `bitgen_t` pointer from `bit_generator`’s capsule. Doesn’t touch the lock.
    ///
    /// # Safety
    ///
    /// The caller must ensure the result has exclusive access to the `bitgen_t` for its whole
    /// lifetime: either `bit_generator` is freshly created and not handed out elsewhere, or its
    /// lock is held (as by [`PyBitGeneratorGuard`]) the entire time.
    unsafe fn new(bit_generator: Bound<'_, PyBitGenerator>) -> PyResult<Self> {
        let py = bit_generator.py();
        let capsule = bit_generator
            .getattr(intern!(py, "capsule"))?
            .cast_into::<PyCapsule>()?;
        let raw = capsule
            .pointer_checked(Some(ffi::c_str!("BitGenerator")))
            .map_err(|_| PyRuntimeError::new_err("Invalid BitGenerator capsule"))?;
        Ok(BitGenerator {
            raw: raw.cast(),
            _bit_generator: bit_generator.unbind(),
        })
    }

    fn addr(&self) -> usize {
        self.raw.as_ptr() as usize
    }

    /// Re-acquire the lock so this generator can be shared with (and used from) Python again.
    pub fn share(self, py: Python<'_>) -> PyResult<PyBitGeneratorGuard> {
        self._bit_generator.bind(py).lock()
    }

    /// Returns the next random unsigned 64 bit integer.
    pub fn next_u64(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_uint64)((*bitgen).state)
        }
    }
    /// Returns the next random unsigned 32 bit integer.
    pub fn next_u32(&mut self) -> u32 {
        unsafe {
            let bitgen = self.raw.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_uint32)((*bitgen).state)
        }
    }
    /// Returns the next random double.
    pub fn next_double(&mut self) -> f64 {
        unsafe {
            let bitgen = self.raw.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_double)((*bitgen).state)
        }
    }
    /// Returns the next raw value (can be used for testing).
    pub fn next_raw(&mut self) -> u64 {
        unsafe {
            let bitgen = self.raw.as_ptr();
            debug_assert_ne!((*bitgen).state, std::ptr::null_mut());
            ((*bitgen).next_raw)((*bitgen).state)
        }
    }
}

#[cfg(feature = "rand_core")]
impl rand_core::RngCore for BitGenerator {
    fn next_u32(&mut self) -> u32 {
        BitGenerator::next_u32(self)
    }
    fn next_u64(&mut self) -> u64 {
        BitGenerator::next_u64(self)
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dst)
    }
}

/// A locked, shared [`PyBitGenerator`], usable without the GIL. Derefs to [`BitGenerator`].
///
/// Since [dropping](`Drop::drop`) this reacquires the GIL,
/// prefer to call [`release`][`PyBitGeneratorGuard::release`] manually to release the lock.
pub struct PyBitGeneratorGuard {
    generator: BitGenerator,
    /// Whether this guard has been manually released.
    released: bool,
    /// This lock makes sure no other threads try to use the BitGenerator while we do.
    /// Since it’s a reentrant `RLock`, `locked_bitgens` closes the same-thread reentrancy gap.
    lock: Py<PyAny>,
}

impl Deref for PyBitGeneratorGuard {
    type Target = BitGenerator;
    fn deref(&self) -> &BitGenerator {
        &self.generator
    }
}

impl DerefMut for PyBitGeneratorGuard {
    fn deref_mut(&mut self) -> &mut BitGenerator {
        &mut self.generator
    }
}

impl Drop for PyBitGeneratorGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        locked_bitgens()
            .lock()
            .unwrap()
            .remove(&self.generator.addr());
        // ignore errors because `drop` can’t fail
        let _ = Python::attach(|py| -> PyResult<_> {
            self.lock.bind(py).call_method0(intern!(py, "release"))?;
            Ok(())
        });
    }
}

impl<'py> PyBitGeneratorGuard {
    /// Release the lock, allowing for checking for errors.
    pub fn release(mut self, py: Python<'py>) -> PyResult<()> {
        self.released = true; // only ever read by drop at the end of a scope (like this one).
        locked_bitgens()
            .lock()
            .unwrap()
            .remove(&self.generator.addr());
        self.lock.bind(py).call_method0(intern!(py, "release"))?;
        Ok(())
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
        use rand::Rng;
        use std::sync::{Arc, Mutex};

        Python::attach(|py| -> PyResult<_> {
            let mut arr = PyArray2::<u32>::zeros(py, (2, 300), false).readwrite();
            let bitgen = get_bit_generator(py)?.lock()?;
            let bitgen = Arc::new(Mutex::new(bitgen));

            let mut arr = arr.as_array_mut();
            py.detach(|| {
                std::thread::scope(|s| {
                    for mut chunk in arr.rows_mut() {
                        let bitgen = Arc::clone(&bitgen);
                        s.spawn(move || {
                            let mut bitgen = bitgen.lock().unwrap();
                            for x in chunk.iter_mut() {
                                *x = bitgen.random_range(10..200);
                            }
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

    /// Python can’t deallocate a PyBitGenerator while it’s locked
    #[test]
    fn lock_keeps_bit_generator_alive() -> PyResult<()> {
        Python::attach(|py| {
            let generator = get_bit_generator(py)?;
            let get_refcount = || {
                py.import("sys")?
                    .call_method1("getrefcount", (&generator,))?
                    .extract::<usize>()
            };

            let refcount_before = get_refcount()?;

            let bitgen = generator.lock()?;
            let refcount_locked = get_refcount()?;
            assert!(refcount_locked > refcount_before);

            bitgen.release(py)?;
            let refcount_after = get_refcount()?;
            assert_eq!(refcount_after, refcount_before);
            Ok(())
        })
    }

    /// Locking a PyBitGenerator twice fails
    #[test]
    fn double_lock_fails_direct() -> PyResult<()> {
        Python::attach(|py| {
            let generator = get_bit_generator(py)?;
            let bitgen = generator.lock()?;
            assert!(generator.lock().is_err());
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }

    /// Locking a bit generator twice fails even if it’s not the same Rust object
    #[test]
    fn double_lock_fails_cloned() -> PyResult<()> {
        Python::attach(|py| {
            let get_bg_ptr = |gen: &Bound<'_, _>| {
                gen.getattr("capsule")?
                    .cast::<PyCapsule>()?
                    .pointer_checked(Some(ffi::c_str!("BitGenerator")))
            };

            let generator1 = get_bit_generator(py)?;
            let generator2 = generator1.clone().into_any();
            let generator2 = generator2.cast::<PyBitGenerator>()?;
            assert_eq!(get_bg_ptr(&generator1)?, get_bg_ptr(&generator2)?);

            let bitgen = generator1.lock()?;
            assert!(generator2.lock().is_err());
            assert!(bitgen.release(py).is_ok());
            Ok(())
        })
    }

    /// Spawned children are independent and owned, so they can be used (and dropped) from their
    /// own threads without locking or a manual `release`.
    #[test]
    fn spawn_produces_independent_generators() -> PyResult<()> {
        Python::attach(|py| {
            let children = get_bit_generator(py)?.spawn(2)?;
            assert_eq!(children.len(), 2);

            let values = py.detach(|| {
                std::thread::scope(|s| {
                    children
                        .into_iter()
                        .map(|mut child| s.spawn(move || child.next_u64()))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .collect::<Vec<_>>()
                })
            });

            assert_ne!(values[0], values[1]);
            Ok(())
        })
    }

    /// A spawned child can be `share`d back into a lockable guard.
    #[test]
    fn spawn_child_can_be_shared() -> PyResult<()> {
        Python::attach(|py| {
            let child = get_bit_generator(py)?.spawn(1)?.pop().unwrap();
            let mut guard = child.share(py)?;
            let _ = py.detach(|| guard.next_u64());
            assert!(guard.release(py).is_ok());
            Ok(())
        })
    }
}

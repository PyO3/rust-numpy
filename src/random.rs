//! Safe interface for NumPy's random [`BitGenerator`][bg].
//!
//! Using the patterns described in [“Extending `numpy.random`”][ext],
//! you can generate random numbers without being attached to the interpreter runtime in one of the following ways:
//! - [lock][`PyBitGeneratorMethods::lock`] a [`PyBitGenerator`] you received from Python
//! - [spawn][`PyBitGeneratorMethods::spawn`] fresh [`BitGenerator`]s from it
//! - obtain a fresh [`BitGenerator`] [from numpy][`BitGenerator::from_numpy`]:
//!
//! ```
//! use pyo3::prelude::*;
//! use numpy::random::BitGenerator;
//!
//! let mut bitgen = Python::attach(|py| {
//!     BitGenerator::from_numpy(py, Default::default())
//! })?;
//! let random_number = bitgen.next_u64();
//! # Ok::<(), PyErr>(())
//! ```
//!
//! If you write a pyo3 extension, you would extract
//! a `numpy.random.BitGenerator` into a [`PyBitGenerator`]:
//!
//! ```
//! # use pyo3::prelude::*;
//! use numpy::random::{BitGenerator, PyBitGenerator, PyBitGeneratorMethods as _};
//! # fn default_bit_gen<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
//! #     Ok(BitGenerator::from_numpy(py, Default::default())?.into_shared().into_bound(py))
//! # }
//!
//! #[pyfunction]
//! fn super_fast_random_number(bitgen: Bound<PyBitGenerator>) -> PyResult<u64> {
//!     let py = bitgen.py();
//!     // lock the generator, then use it without being attached to the interpreter runtime
//!     bitgen.lock(|bitgen| py.detach(|| bitgen.next_u64()))
//! }
//!
//! Python::attach(|py| -> PyResult<_> {
//!     let bitgen: Bound<PyBitGenerator> = default_bit_gen(py)?;
//!     let random_number = super_fast_random_number(bitgen)?;
//!     println!("{random_number}");
//!     Ok(())
//! })?;
//! # Ok::<(), PyErr>(())
//! ```
//!
//! With the `rand` crate installed, you can also use its `Rng` APIs on any generator,
//! since [`BitGenerator`] implements [`rand_core::RngCore`].
//!
//! ```
//! use pyo3::prelude::*;
//! use rand::Rng as _;
//! use numpy::random::{BitGenerator, NumpyBitGenerator::SFC64};
//!
//! let mut bitgen = Python::attach(|py| BitGenerator::from_numpy(py, SFC64))?;
//! if bitgen.random_ratio(1, 1_000_000) {
//!     println!("a sure thing");
//! };
//! # Ok::<(), PyErr>(())
//! ```
//!
//! Using `spawn`, you can create multiple [`BitGenerator`]s to generate random numbers truly in parallel,
//! all without being attached to the interpreter runtime:
//!
//! ```
//! # use pyo3::prelude::*;
//! # use rand::Rng as _;
//! use numpy::{PyArray2, PyArrayMethods as _};
//! # use numpy::random::{BitGenerator, PyBitGenerator, PyBitGeneratorMethods as _};
//! # fn default_bit_gen<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
//! #     Ok(BitGenerator::from_numpy(py, Default::default())?.into_shared().into_bound(py))
//! # }
//!
//! Python::attach(|py| -> PyResult<_> {
//!     let bitgen: Bound<PyBitGenerator> = default_bit_gen(py)?;
//!     let children = bitgen.spawn(4)?;
//!     let mut arr = PyArray2::<u32>::zeros(py, (4, 300), false).readwrite();
//!     let mut ndarr = arr.as_array_mut();  // ndarray for more convenience
//!     py.detach(|| std::thread::scope(|s| {
//!         for (mut chunk, mut child) in ndarr.rows_mut().into_iter().zip(children) {
//!             s.spawn(move || {
//!                 for x in chunk.iter_mut() {
//!                     *x = child.random_range(10..200);
//!                 }
//!             });
//!         }
//!     }));
//!     println!("Now filled: {arr:?}");
//!     Ok(())
//! })?;
//! # Ok::<(), PyErr>(())
//! ```
//!
//! [bg]: https://numpy.org/doc/stable//reference/random/bit_generators/generated/numpy.random.BitGenerator.html
//! [ext]: https://numpy.org/doc/stable/reference/random/extending.html

use std::cell::RefCell;
use std::collections::HashSet;
use std::ptr::NonNull;

use pyo3::{
    exceptions::PyRuntimeError,
    ffi, intern,
    prelude::*,
    sync::PyOnceLock,
    types::{DerefToPyAny, PyCapsule, PyType},
    PyTypeInfo,
};

use crate::npyffi::bitgen_t;

/// Methods for [`PyBitGenerator`].
pub trait PyBitGeneratorMethods {
    /// Lock the bit generator, run `f` with exclusive access to it,
    /// then release the lock (even on panic).
    /// `f` may use it without being attached to the interpreter runtime via [`Python::detach`].
    fn lock<R>(&self, f: impl FnOnce(&mut BitGenerator) -> R) -> PyResult<R>;

    /// Spawn `n_children` independent child [`BitGenerator`]s.
    ///
    /// This is the recommended way to obtain generators for multiple threads: unlike sharing a
    /// single [locked][PyBitGeneratorMethods::lock] one, each child has its own, independent state.
    fn spawn(&self, n_children: usize) -> PyResult<Vec<BitGenerator>>;

    /// Spawn a single owned child [`BitGenerator`] that doesn’t need locking to be used.
    fn spawn_one(&self) -> PyResult<BitGenerator> {
        self.spawn(1).map(|mut v| v.pop().unwrap())
    }
}

thread_local! {
    /// Addresses of the `bitgen_t`s currently locked on this thread.
    /// `BitGenerator.lock` is a reentrant `RLock` preventing cross-thread use,
    /// and this helps rejecting same thread re-locking.
    static LOCKED: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
}

struct SameThreadLockGuard<'py> {
    lock: Bound<'py, PyAny>,
    addr: usize,
    release_on_drop: bool,
}

/// Release the `BitGenerator` lock and clear its reentrancy marker, even on panic.
impl<'py> SameThreadLockGuard<'py> {
    /// Takes a locked lock, and returns a `LockGuard` that releases it on drop or `release`.
    fn new(lock: Bound<'py, PyAny>, generator: &BitGenerator) -> PyResult<Self> {
        let addr = generator.addr();
        if !LOCKED.with_borrow_mut(|locked| locked.insert(addr)) {
            lock.call_method0(intern!(lock.py(), "release"))?;
            return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
        }
        Ok(Self {
            lock,
            addr,
            release_on_drop: true,
        })
    }
    fn release(&mut self) -> PyResult<()> {
        self.release_on_drop = false;
        LOCKED.with_borrow_mut(|locked| locked.remove(&self.addr));
        self.lock.call_method0(intern!(self.lock.py(), "release"))?;
        Ok(())
    }
}

impl Drop for SameThreadLockGuard<'_> {
    fn drop(&mut self) {
        if self.release_on_drop {
            // ignore errors because `drop` can’t fail
            let _ = self.release();
        }
    }
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

impl<'py> PyBitGeneratorMethods for Bound<'py, PyBitGenerator> {
    fn lock<R>(&self, f: impl FnOnce(&mut BitGenerator) -> R) -> PyResult<R> {
        let py = self.py();
        let lock = self.getattr(intern!(py, "lock"))?;
        // Acquire the (reentrant!) lock in non-blocking mode or return an error.
        if !lock
            .call_method1(intern!(py, "acquire"), (false,))?
            .extract()?
        {
            return Err(PyRuntimeError::new_err(
                "Failed to acquire BitGenerator lock",
            ));
        }
        // SAFETY: we hold the lock until the end of this scope (the `LockGuard` releases it),
        //         and reject reentrant re-locking below, so `generator`’s access stays exclusive.
        let mut generator = match unsafe { BitGenerator::from_py(self.clone()) } {
            Ok(generator) => generator,
            Err(err) => {
                lock.call_method0(intern!(py, "release"))?;
                return Err(err);
            }
        };
        let mut guard = SameThreadLockGuard::new(lock, &generator)?;
        let rv = f(&mut generator);
        guard.release()?; // `f` didn’t panic, so we can release the lock fallibly here.
        Ok(rv)
    }

    fn spawn(&self, n_children: usize) -> PyResult<Vec<BitGenerator>> {
        let py = self.py();
        self.call_method1(intern!(py, "spawn"), (n_children,))?
            .try_iter()?
            // SAFETY: each child is freshly spawned and only handed to us, so it’s exclusively ours.
            .map(|child| unsafe { BitGenerator::from_py(child?.cast_into::<PyBitGenerator>()?) })
            .collect()
    }
}

/// Known numpy bit generator types.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumpyBitGenerator {
    /// Mersenne Twister (MT19937)
    MT19937,
    /// Permuted congruential generator (64-bit, PCG-64)
    #[default]
    PCG64,
    /// Permuted congruential generator (64-bit, PCG-64 DXSM
    PCG64DXSM,
    /// Philox counter-based RNG
    Philox,
    /// SFC64 Small Fast Chaotic PRNG
    SFC64,
}

impl From<NumpyBitGenerator> for &'static str {
    fn from(value: NumpyBitGenerator) -> &'static str {
        match value {
            NumpyBitGenerator::MT19937 => "MT19937",
            NumpyBitGenerator::PCG64 => "PCG64",
            NumpyBitGenerator::PCG64DXSM => "PCG64dxsm",
            NumpyBitGenerator::Philox => "Philox",
            NumpyBitGenerator::SFC64 => "SFC64",
        }
    }
}

/// A numpy `BitGenerator` usable without being attached to the interpreter runtime,
/// with exclusive access to its state.
///
/// [`spawn`][PyBitGeneratorMethods::spawn] hands out independent, owned ones;
/// [`lock`][PyBitGeneratorMethods::lock] passes one borrowing a shared generator under its lock.
pub struct BitGenerator {
    raw: NonNull<bitgen_t>,
    /// Keeps `raw` alive: the capsule’s pointer lives in memory owned by the `BitGenerator`, which
    /// has no back-reference of its own, so only keeping it alive keeps that memory valid.
    _bit_generator: Py<PyBitGenerator>,
}

// SAFETY: `raw` is only ever accessed through `&mut self`, so it can’t be used in parallel, and we
//         keep its `bitgen_t` alive via `_bit_generator`. Every `BitGenerator` has exclusive access
//         to its `bitgen_t` (a fresh `spawn` child it owns, or a shared one protected by a held lock
//         for the duration of a [`lock`][PyBitGeneratorMethods::lock] call), so nothing else can
//         touch its state.
unsafe impl Send for BitGenerator {}

impl BitGenerator {
    /// Creates a fresh [`BitGenerator`] backed by numpy’s implementation.
    ///
    /// ```
    /// use pyo3::prelude::*;
    /// use numpy::random::{BitGenerator, NumpyBitGenerator};
    ///
    /// let mut bitgen = Python::attach(|py| BitGenerator::from_numpy(py, Default::default()))?;
    /// println!("{}", bitgen.next_u32());
    /// # Ok::<(), PyErr>(())
    /// ```
    pub fn from_numpy(py: Python<'_>, flavor: NumpyBitGenerator) -> PyResult<Self> {
        let bitgen = py
            .import("numpy.random")?
            .call_method0::<&str>(flavor.into())?
            .cast_into::<PyBitGenerator>()?;
        // SAFETY: `bitgen` is freshly created and not handed out elsewhere.
        unsafe { Self::from_py(bitgen) }
    }

    /// Extracts the raw `bitgen_t` pointer from `bit_generator`’s capsule. Doesn’t touch the lock.
    ///
    /// # Safety
    ///
    /// The caller must ensure the result has exclusive access to the `bitgen_t` for its whole lifetime:
    /// either `bit_generator` is freshly created and not handed out elsewhere,
    /// or its lock is held (as by [`lock`][PyBitGeneratorMethods::lock]) the entire time.
    unsafe fn from_py(bit_generator: Bound<'_, PyBitGenerator>) -> PyResult<Self> {
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

    /// Returns the underlying [`PyBitGenerator`].
    pub fn into_shared(self) -> Py<PyBitGenerator> {
        self._bit_generator
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

    /// Simple single-threaded use: lock the generator,
    /// then use it without being attached to the interpreter runtime.
    #[test]
    fn use_detached() -> PyResult<()> {
        Python::attach(|py| {
            get_bit_generator(py)?.lock(|bitgen| {
                py.detach(|| {
                    let _ = bitgen.next_raw();
                });
            })
        })
    }

    /// Use single shared generator from multiple threads (not very useful but possible)
    #[cfg(feature = "rand_core")]
    #[test]
    fn use_concurrent() -> PyResult<()> {
        use crate::array::{PyArray2, PyArrayMethods as _};
        use rand::Rng;
        use std::sync::Mutex;

        Python::attach(|py| -> PyResult<_> {
            let mut arr = PyArray2::<u32>::zeros(py, (2, 300), false).readwrite();
            let mut arr = arr.as_array_mut();
            get_bit_generator(py)?.lock(|bitgen| {
                let bitgen = Mutex::new(bitgen);
                py.detach(|| {
                    std::thread::scope(|s| {
                        for mut chunk in arr.rows_mut() {
                            let bitgen = &bitgen;
                            s.spawn(move || {
                                let mut bitgen = bitgen.lock().unwrap();
                                for x in chunk.iter_mut() {
                                    *x = bitgen.random_range(10..200);
                                }
                            });
                        }
                    })
                });
            })
        })
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand_core")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        Python::attach(|py| {
            get_bit_generator(py)?.lock(|bitgen| {
                py.detach(|| {
                    assert!(bitgen.random_ratio(1, 1));
                    assert!(!bitgen.random_ratio(0, 1));
                });
            })
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
            generator.lock(|_| {
                assert!(get_refcount().unwrap() > refcount_before);
            })?;
            assert_eq!(get_refcount()?, refcount_before);
            Ok(())
        })
    }

    /// Locking a PyBitGenerator while it’s already locked fails
    #[test]
    fn double_lock_fails_direct() -> PyResult<()> {
        Python::attach(|py| {
            let generator = get_bit_generator(py)?;
            generator.lock(|_| {
                assert!(generator.lock(|_| {}).is_err());
            })
        })
    }

    /// Locking a bit generator twice fails even through a separate handle obtained from Python
    #[test]
    fn double_lock_fails_roundtrip() -> PyResult<()> {
        use pyo3::types::PyList;

        Python::attach(|py| {
            let get_bg_ptr = |gen: &Bound<'_, _>| {
                gen.getattr("capsule")?
                    .cast::<PyCapsule>()?
                    .pointer_checked(Some(ffi::c_str!("BitGenerator")))
            };

            let generator1 = get_bit_generator(py)?;
            // round-trip the very same object through Python, so no Rust-side cloning is involved
            let generator2 = PyList::new(py, [&generator1])?
                .get_item(0)?
                .cast_into::<PyBitGenerator>()?;
            assert_eq!(get_bg_ptr(&generator1)?, get_bg_ptr(&generator2)?);

            generator1.lock(|_| {
                assert!(generator2.lock(|_| {}).is_err());
            })
        })
    }

    /// Spawned children are independent and owned,
    /// so they can be used (and dropped) from their own threads without locking.
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
}

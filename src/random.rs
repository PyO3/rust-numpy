//! Safe interface for NumPy's random [`BitGenerator`][bg].
//!
//! Using the patterns described in [“Extending `numpy.random`”][ext],
//! you can generate random numbers without being attached to the interpreter runtime in one of the following ways:
//! - [lock][`PyBitGeneratorMethods::lock`] a [`PyBitGenerator`] you received from Python
//! - [spawn][`PyBitGeneratorMethods::spawn`] fresh [`BitGenerator`]s from it
//! - create a fresh [`BitGenerator`] [from numpy][`BitGenerator::new`]:
//!
//! ```
//! use pyo3::prelude::*;
//! use numpy::random::BitGenerator;
//!
//! let mut bitgen = Python::attach(|py| {
//!     BitGenerator::new(py, Default::default())
//! })?;
//! let random_number = bitgen.next_u64();
//! # Ok::<(), PyErr>(())
//! ```
//!
//! If you write a pyo3 extension, you would extract
//! a [`numpy.random.BitGenerator`] into a <code>[Bound]<'_, [PyBitGenerator]></code>:
//!
//! [`numpy.random.BitGenerator`]: https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.BitGenerator.html
//!
//! ```
//! # use pyo3::prelude::*;
//! use numpy::random::{BitGenerator, PyBitGenerator, PyBitGeneratorMethods as _};
//! # fn default_bit_gen<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
//! #     Ok(BitGenerator::new(py, Default::default())?.into_shared().into_bound(py))
//! # }
//!
//! #[pyfunction]
//! fn super_fast_random_number(bitgen: Bound<PyBitGenerator>) -> PyResult<u64> {
//!     // lock the generator, then use it without being attached to the interpreter runtime
//!     bitgen.lock(|mut bitgen| bitgen.next_u64())
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
//! use numpy::random::{BitGenerator, BitGeneratorKind::SFC64};
//!
//! let mut bitgen = Python::attach(|py| BitGenerator::new(py, SFC64))?;
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
//! #     Ok(BitGenerator::new(py, Default::default())?.into_shared().into_bound(py))
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

use std::collections::HashSet;
use std::ptr::NonNull;
use std::{cell::RefCell, mem::ManuallyDrop};

use pyo3::{
    exceptions::PyRuntimeError,
    ffi, intern,
    marker::Ungil,
    prelude::*,
    sync::PyOnceLock,
    types::{DerefToPyAny, PyCapsule, PyType},
    PyTypeInfo,
};

use crate::npyffi::bitgen_t;

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

/// Methods for [`PyBitGenerator`].
pub trait PyBitGeneratorMethods: Sealed {
    /// Lock the bit generator, run `f` with exclusive access to it,
    /// then release the lock (even on panic).
    /// `f` may use it without being attached to the interpreter runtime via [`Python::detach`].
    fn lock<R: Ungil>(
        &self,
        f: impl (FnOnce(BitGeneratorRef<'_>) -> R) + Ungil + Send,
    ) -> PyResult<R>;

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

mod guard {
    use super::*;

    pub(super) struct SameThreadLockGuard<'py> {
        lock: Bound<'py, PyAny>,
        addr: usize,
    }

    /// Release the `BitGenerator` lock and clear its reentrancy marker, even on panic.
    impl<'py> SameThreadLockGuard<'py> {
        /// Takes a locked lock, and returns a `LockGuard` that releases it on drop or `release`.
        pub(super) fn new(lock: Bound<'py, PyAny>, generator: &BitGenerator) -> PyResult<Self> {
            let addr = generator.addr();
            if !LOCKED.with_borrow_mut(|locked| locked.insert(addr)) {
                lock.call_method0(intern!(lock.py(), "release"))?;
                return Err(PyRuntimeError::new_err("BitGenerator is already locked"));
            }
            Ok(Self { lock, addr })
        }
        /// Releases the lock.
        pub(super) fn release(self) -> PyResult<()> {
            let mut s = ManuallyDrop::new(self);
            s.release_borrowed()?;
            Ok(())
        }
        fn release_borrowed(&mut self) -> PyResult<()> {
            LOCKED.with_borrow_mut(|locked| locked.remove(&self.addr));
            self.lock.call_method0(intern!(self.lock.py(), "release"))?;
            Ok(())
        }
    }

    impl Drop for SameThreadLockGuard<'_> {
        fn drop(&mut self) {
            // ignore errors because `drop` can’t fail
            let _ = self.release_borrowed();
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
    fn lock<R: Ungil>(
        &self,
        f: impl (FnOnce(BitGeneratorRef<'_>) -> R) + Ungil + Send,
    ) -> PyResult<R> {
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
        let guard = guard::SameThreadLockGuard::new(lock, &generator)?;
        let rv = py.detach(|| f(BitGeneratorRef(&mut generator)));
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

impl Sealed for Bound<'_, PyBitGenerator> {}

/// Which of numpy’s bit generator algorithms [`BitGenerator::new`] should create.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitGeneratorKind {
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

impl From<BitGeneratorKind> for &'static str {
    fn from(value: BitGeneratorKind) -> &'static str {
        match value {
            BitGeneratorKind::MT19937 => "MT19937",
            BitGeneratorKind::PCG64 => "PCG64",
            BitGeneratorKind::PCG64DXSM => "PCG64DXSM",
            BitGeneratorKind::Philox => "Philox",
            BitGeneratorKind::SFC64 => "SFC64",
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
    /// use numpy::random::{BitGenerator, BitGeneratorKind};
    ///
    /// let mut bitgen = Python::attach(|py| BitGenerator::new(py, Default::default()))?;
    /// println!("{}", bitgen.next_u32());
    /// # Ok::<(), PyErr>(())
    /// ```
    pub fn new(py: Python<'_>, kind: BitGeneratorKind) -> PyResult<Self> {
        let bitgen = py
            .import("numpy.random")?
            .call_method0::<&str>(kind.into())?
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

/// Exclusive access to a *shared* [`BitGenerator`], handed to the closure of
/// [`lock`][PyBitGeneratorMethods::lock] for as long as its lock is held.
///
/// It deliberately doesn’t hand out `&mut BitGenerator`: that would let the closure
/// [swap][std::mem::swap] the shared generator out and keep using it after the lock is released:
///
/// ```compile_fail
/// # use pyo3::prelude::*;
/// # use numpy::random::{BitGenerator, BitGeneratorKind, PyBitGenerator, PyBitGeneratorMethods as _};
/// # fn shared<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
/// #     Ok(BitGenerator::new(py, Default::default())?.into_shared().into_bound(py))
/// # }
/// Python::attach(|py| {
///     let mut mine = BitGenerator::new(py, BitGeneratorKind::PCG64)?;
///     // `shared` is a `BitGeneratorRef`, not a `&mut BitGenerator`, so this doesn’t compile:
///     shared(py)?.lock(|shared| std::mem::swap(shared, &mut mine))?;
///     mine.next_double(); // would be unsynchronized access to the shared generator
///     Ok::<(), PyErr>(())
/// })?;
/// # Ok::<(), PyErr>(())
/// ```
pub struct BitGeneratorRef<'a>(&'a mut BitGenerator);

impl BitGeneratorRef<'_> {
    /// See [`BitGenerator::next_u64`].
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    /// See [`BitGenerator::next_u32`].
    pub fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    /// See [`BitGenerator::next_double`].
    pub fn next_double(&mut self) -> f64 {
        self.0.next_double()
    }
    /// See [`BitGenerator::next_raw`].
    pub fn next_raw(&mut self) -> u64 {
        self.0.next_raw()
    }
}

#[cfg(feature = "rand_core")]
impl rand_core::RngCore for BitGeneratorRef<'_> {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dst)
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
        let bit_generator = py
            .import("numpy.random")?
            .call_method1("PCG64", (42,))?
            .cast_into::<PyBitGenerator>()?;
        Ok(bit_generator)
    }

    /// Simple single-threaded use: lock the generator,
    /// then use it without being attached to the interpreter runtime.
    #[test]
    fn use_detached() -> PyResult<()> {
        Python::attach(|py| {
            get_bit_generator(py)?.lock(|mut bitgen| {
                assert_eq!(bitgen.next_raw(), 14276969152011380360);
            })
        })
    }

    #[test]
    fn use_owned() -> PyResult<()> {
        let mut bitgen = Python::attach(|py| get_bit_generator(py)?.spawn_one())?;
        assert_eq!(bitgen.next_raw(), 16910944855483863638);
        Ok(())
    }

    /// Use single shared generator from multiple threads (not very useful but possible)
    #[cfg(feature = "rand_core")]
    #[test]
    fn use_concurrent() -> PyResult<()> {
        use crate::array::{PyArray2, PyArrayMethods as _};
        use rand::Rng;
        use std::sync::Mutex;

        Python::attach(|py| -> PyResult<_> {
            let mut arr = PyArray2::<u32>::zeros(py, (2, 3), false).readwrite();
            let mut arr = arr.as_array_mut();
            get_bit_generator(py)?.lock(|bitgen| {
                let bitgen = Mutex::new(bitgen);
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
            })?;
            assert_eq!(arr, ndarray::array![[26, 157, 134], [93, 92, 173]]);
            Ok(())
        })
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand_core")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        Python::attach(|py| {
            get_bit_generator(py)?.lock(|mut bitgen| {
                // check reproducibility
                let seq: Vec<bool> = std::iter::repeat_with(|| bitgen.random_ratio(1, 2))
                    .take(10)
                    .collect();
                assert_eq!(
                    seq,
                    vec![false, true, false, false, true, false, false, false, true, true]
                );

                // check trivial ratios
                for _ in 0..100 {
                    assert!(bitgen.random_ratio(1, 1));
                    assert!(!bitgen.random_ratio(0, 1));
                }
            })
        })
    }

    /// Re-locking the same generator on the same thread is rejected,
    /// even where numpy’s `lock` is a reentrant `RLock` (numpy ≥ 2.4) and would allow it.
    #[test]
    fn reject_reentrant_lock() -> PyResult<()> {
        Python::attach(|py| {
            let bit_generator = get_bit_generator(py)?;
            let shared = bit_generator.clone().unbind();
            let err = bit_generator
                .lock(|_| Python::attach(|py| shared.bind(py).lock(|_| ()).unwrap_err()))?;
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(
                matches!(
                    err.value(py).to_string().as_str(),
                    // numpy < 2.4 uses a plain `Lock`, so there `acquire` fails before our check
                    "BitGenerator is already locked" | "Failed to acquire BitGenerator lock"
                ),
                "unexpected error: {err}"
            );
            // the lock and the reentrancy marker were cleaned up, so locking again works
            bit_generator.lock(|mut bitgen| bitgen.next_u64())?;
            Ok(())
        })
    }

    /// A panicking closure still releases the lock and clears the reentrancy marker
    /// (via the guard’s `Drop`), so the generator stays usable.
    #[test]
    fn release_lock_on_panic() -> PyResult<()> {
        Python::attach(|py| {
            let bit_generator = get_bit_generator(py)?;
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bit_generator.lock(|_| panic!("boom"))
            }))
            .unwrap_err();
            assert_eq!(panic.downcast_ref::<&str>(), Some(&"boom"));

            assert_eq!(
                bit_generator.lock(|mut bitgen| bitgen.next_raw())?,
                14276969152011380360
            );
            Ok(())
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

            assert_eq!(values, vec![16910944855483863638, 8623682774590505111]);
            Ok(())
        })
    }
}

//! Safe interface for NumPy's random [`BitGenerator`][bg].
//!
//! Using the patterns described in [“Extending `numpy.random`”][ext],
//! you can generate random numbers without being attached to the interpreter runtime by:
//! - [spawning][`PyBitGeneratorMethods::spawn`] fresh [`BitGenerator`]s
//!   from a [`PyBitGenerator`] you received from Python
//! - creating a fresh [`BitGenerator`] [from numpy][`BitGenerator::new`]:
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
//! fn make_random_number(bitgen: Bound<PyBitGenerator>) -> PyResult<u64> {
//!     // spawn an owned child, then use it without being attached to the interpreter runtime
//!     Ok(bitgen.spawn_one()?.next_u64())
//! }
//!
//! Python::attach(|py| -> PyResult<_> {
//!     let bitgen: Bound<PyBitGenerator> = default_bit_gen(py)?;
//!     let random_number = make_random_number(bitgen)?;
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

mod sealed {
    pub trait Sealed {}
}

use sealed::Sealed;

/// Methods for [`PyBitGenerator`].
pub trait PyBitGeneratorMethods: Sealed {
    /// Spawn `n_children` independent child [`BitGenerator`]s.
    ///
    /// This is the way to obtain generators for multiple threads: each child has its own,
    /// independent state, so no synchronization is needed between them.
    fn spawn(&self, n_children: usize) -> PyResult<Vec<BitGenerator>>;

    /// Spawn a single owned child [`BitGenerator`].
    fn spawn_one(&self) -> PyResult<BitGenerator> {
        let mut children = self.spawn(1)?;
        children
            .pop()
            .ok_or_else(|| PyRuntimeError::new_err("spawn(1) returned no children"))
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

impl BitGeneratorKind {
    /// Returns an iterator over the values of [`BitGeneratorKind`].
    pub fn iter() -> std::iter::Copied<std::slice::Iter<'static, BitGeneratorKind>> {
        use BitGeneratorKind::*;
        static KINDS: [BitGeneratorKind; 5] = [MT19937, PCG64, PCG64DXSM, Philox, SFC64];
        KINDS.iter().copied()
    }
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
/// [`spawn`][PyBitGeneratorMethods::spawn] hands out independent, owned ones.
pub struct BitGenerator {
    raw: NonNull<bitgen_t>,
    /// Keeps `raw` alive: the capsule’s pointer lives in memory owned by the `BitGenerator`, which
    /// has no back-reference of its own, so only keeping it alive keeps that memory valid.
    _bit_generator: Py<PyBitGenerator>,
}

// SAFETY: `raw` is only ever accessed through `&mut self`, so it can’t be used in parallel, and we
//         keep its `bitgen_t` alive via `_bit_generator`. Every `BitGenerator` owns its `bitgen_t`
//         exclusively (it is freshly created or a fresh `spawn` child), so nothing else can touch
//         its state.
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

    /// Extracts the raw `bitgen_t` pointer from `bit_generator`’s capsule.
    ///
    /// # Safety
    ///
    /// The caller must ensure the result has exclusive access to the `bitgen_t` for its whole
    /// lifetime, i.e. `bit_generator` is freshly created and not handed out elsewhere.
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

    fn get_shared<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBitGenerator>> {
        let bitgen = py
            .import("numpy.random")?
            .call_method1("PCG64", (42,))?
            .cast_into::<PyBitGenerator>()?;
        Ok(bitgen)
    }

    fn get_owned<'py>(py: Python<'py>) -> PyResult<BitGenerator> {
        let bitgen = get_shared(py)?;
        // SAFETY: `bitgen` is freshly created and not handed out elsewhere.
        unsafe { BitGenerator::from_py(bitgen) }
    }

    #[test]
    fn from_kind() -> PyResult<()> {
        Python::attach(|py| {
            for kind in BitGeneratorKind::iter() {
                let name: &str = kind.into();
                let type_name = BitGenerator::new(py, kind)?
                    .into_shared()
                    .bind(py)
                    .get_type()
                    .name()?;
                assert_eq!(type_name, name);
            }
            Ok(())
        })
    }

    /// Simple single-threaded use of an owned generator.
    #[test]
    fn base_api() -> PyResult<()> {
        Python::attach(|py| {
            let double = get_owned(py)?.next_double();
            assert_eq!(double, 0.7739560485559633);

            let u32_owned = get_owned(py)?.next_u32();
            assert_eq!(u32_owned, 383329928);

            let u64_owned = get_owned(py)?.next_u64();
            assert_eq!(u64_owned, 14276969152011380360);

            let raw_owned = get_owned(py)?.next_raw();
            assert_eq!(raw_owned, u64_owned);

            Ok(())
        })
    }

    /// Test that the `rand::Rng` APIs work
    #[cfg(feature = "rand_core")]
    #[test]
    fn rand() -> PyResult<()> {
        use rand::Rng as _;

        Python::attach(|py| {
            let seq_owned: Vec<bool> = get_owned(py)?
                .sample_iter(rand::distr::Bernoulli::new(0.5).unwrap())
                .take(10)
                .collect();
            let seq_expected = vec![
                false, true, false, false, true, false, false, false, true, true,
            ];
            assert_eq!(&seq_owned, &seq_expected);
            Ok(())
        })
    }

    #[test]
    fn spawn_one() -> PyResult<()> {
        let mut bitgen = Python::attach(|py| get_shared(py)?.spawn_one())?;
        assert_eq!(bitgen.next_u32(), 2136330838);
        Ok(())
    }

    /// Spawned children are independent and owned,
    /// so they can be used (and dropped) from their own threads without locking.
    #[test]
    fn spawn_produces_independent_generators() -> PyResult<()> {
        Python::attach(|py| {
            let children = get_shared(py)?.spawn(2)?;
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

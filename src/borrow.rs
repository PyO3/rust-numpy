//! Types to safely create references into NumPy arrays
//!
//! It is assumed that unchecked code - which includes unsafe Rust and Python - is validated by its author
//! which together with the dynamic borrow checking performed by this crate ensures that
//! safe Rust code cannot cause undefined behaviour by creating references into NumPy arrays.
//!
//! With these borrows established, [references to individual elements][PyReadonlyArray::get] or [reference-based views of whole array][PyReadonlyArray::as_array]
//! can be created safely. These are then the starting point for algorithms iteraing over and operating on the elements of the array.
//!
//! # Examples
//!
//! The first example shows that dynamic borrow checking works to constrain
//! both what safe Rust code can invoke and how it is invoked.
//!
//! ```rust
//! # use std::panic::{catch_unwind, AssertUnwindSafe};
//! #
//! use numpy::PyArray1;
//! use ndarray::Zip;
//! use pyo3::Python;
//!
//! fn add(x: &PyArray1<f64>, y: &PyArray1<f64>, z: &PyArray1<f64>) {
//!     let x1 = x.readonly();
//!     let y1 = y.readonly();
//!     let mut z1 = z.readwrite();
//!
//!     let x2 = x1.as_array();
//!     let y2 = y1.as_array();
//!     let z2 = z1.as_array_mut();
//!
//!     Zip::from(x2)
//!         .and(y2)
//!         .and(z2)
//!         .for_each(|x3, y3, z3| *z3 = x3 + y3);
//!
//!     // Will fail at runtime due to conflict with `x1`.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _x4 = x.readwrite();
//!     }));
//!     assert!(res.is_err());
//! }
//!
//! Python::with_gil(|py| {
//!     let x = PyArray1::<f64>::zeros(py, 42, false);
//!     let y = PyArray1::<f64>::zeros(py, 42, false);
//!     let z = PyArray1::<f64>::zeros(py, 42, false);
//!
//!     // Will work as the three arrays are distinct.
//!     add(x, y, z);
//!
//!     // Will work as `x1` and `y1` are compatible borrows.
//!     add(x, x, z);
//!
//!     // Will fail at runtime due to conflict between `y1` and `z1`.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         add(x, y, y);
//!     }));
//!     assert!(res.is_err());
//! });
//! ```
//!
//! The second example shows that non-overlapping and interleaved views are also supported.
//!
//! ```rust
//! use numpy::PyArray1;
//! use pyo3::{types::IntoPyDict, Python};
//!
//! Python::with_gil(|py| {
//!     let array = PyArray1::arange(py, 0.0, 10.0, 1.0);
//!     let locals = [("array", array)].into_py_dict(py);
//!
//!     let view1 = py.eval("array[:5]", None, Some(locals)).unwrap().downcast::<PyArray1<f64>>().unwrap();
//!     let view2 = py.eval("array[5:]", None, Some(locals)).unwrap().downcast::<PyArray1<f64>>().unwrap();
//!     let view3 = py.eval("array[::2]", None, Some(locals)).unwrap().downcast::<PyArray1<f64>>().unwrap();
//!     let view4 = py.eval("array[1::2]", None, Some(locals)).unwrap().downcast::<PyArray1<f64>>().unwrap();
//!
//!     {
//!         let _view1 = view1.readwrite();
//!         let _view2 = view2.readwrite();
//!     }
//!
//!     {
//!         let _view3 = view3.readwrite();
//!         let _view4 = view4.readwrite();
//!     }
//! });
//! ```
//!
//! The third example shows that some views are incorrectly rejected since the borrows are over-approximated.
//!
//! ```rust
//! # use std::panic::{catch_unwind, AssertUnwindSafe};
//! #
//! use numpy::PyArray2;
//! use pyo3::{types::IntoPyDict, Python};
//!
//! Python::with_gil(|py| {
//!     let array = PyArray2::<f64>::zeros(py, (10, 10), false);
//!     let locals = [("array", array)].into_py_dict(py);
//!
//!     let view1 = py.eval("array[:, ::3]", None, Some(locals)).unwrap().downcast::<PyArray2<f64>>().unwrap();
//!     let view2 = py.eval("array[:, 1::3]", None, Some(locals)).unwrap().downcast::<PyArray2<f64>>().unwrap();
//!
//!     // A false conflict as the views do not actually share any elements.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _view1 = view1.readwrite();
//!         let _view2 = view2.readwrite();
//!     }));
//!     assert!(res.is_err());
//! });
//! ```
//!
//! # Rationale
//!
//! Rust references require aliasing discipline to be maintained, i.e. there must always
//! exist only a single mutable (aka exclusive) reference or multiple immutable (aka shared) references
//! for each object, otherwise the program contains undefined behaviour.
//!
//! The aim of this module is to ensure that safe Rust code is unable to violate these requirements on its own.
//! We cannot prevent unchecked code - this includes unsafe Rust, Python or other native code like C or Fortran -
//! from violating them. Therefore the responsibility to avoid this lies with the author of that code instead of the compiler.
//! However, assuming that the unchecked code is correct, we can ensure that safe Rust is unable to introduce mistakes
//! into an otherwise correct program by dynamically checking which arrays are currently borrowed and in what manner.
//!
//! This means that we follow the [base object chain][base] of each array to the original allocation backing it and
//! track which parts of that allocation are covered by the array and thereby ensure that only a single read-write array
//! or multiple read-only arrays overlapping with that region are borrowed at any time.
//!
//! In contrast to Rust references, the mere existence of Python references or raw pointers is not an issue
//! because these values are not assumed to follow aliasing discipline by the Rust compiler.
//!
//! This cannot prevent unchecked code from concurrently modifying an array via callbacks or using multiple threads,
//! but that would lead to incorrect results even if the code that is interfered with is implemented in another language
//! which does not require aliasing discipline.
//!
//! Concerning multi-threading in particular: While the GIL needs to be acquired to create borrows, they are not bound to the GIL
//! and will stay active after the GIL is released, for example by calling [`allow_threads`][pyo3::Python::allow_threads].
//! Borrows also do not provide synchronization, i.e. multiple threads borrowing the same array will lead to runtime panics,
//! it will not block those threads until already active borrows are released.
//!
//! In summary, this crate takes the position that all unchecked code - unsafe Rust, Python, C, Fortran, etc. - must be checked for correctness by its author.
//! Safe Rust code can then rely on this correctness, but should not be able to introduce memory safety issues on its own. Additionally, dynamic borrow checking
//! can catch _some_ mistakes introduced by unchecked code, e.g. Python calling a function with the same array as an input and as an output argument.
//!
//! # Limitations
//!
//! Note that the current implementation of this is an over-approximation: It will consider borrows
//! potentially conflicting if the initial arrays have the same object at the end of their [base object chain][base].
//! Then, multiple conditions which are sufficient but not necessary to show the absence of conflicts are checked.
//!
//! While this is sufficient to handle common situations like slicing an array with a non-unit step size which divides
//! the dimension along that axis, there are also cases which it does not handle. For example, if the step size does
//! not divide the dimension along the sliced axis. Under such conditions, borrows are rejected even though the arrays
//! do not actually share any elements.
//!
//! This does limit the set of programs that can be written using safe Rust in way similar to rustc itself
//! which ensures that all accepted programs are memory safe but does not necessarily accept all memory safe programs.
//! However, the unsafe method [`PyArray::as_array_mut`] can be used as an escape hatch.
//! More involved cases like the example from above may be supported in the future.
//!
//! [base]: https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.NPY_AO.base

use std::any::type_name;
use std::cell::UnsafeCell;
use std::collections::hash_map::Entry;
use std::fmt;
use std::mem::size_of;
use std::ops::Deref;

use ndarray::{
    ArrayView, ArrayViewMut, Dimension, IntoDimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
};
use num_integer::gcd;
use pyo3::{FromPyObject, PyAny, PyResult, Python};
use rustc_hash::FxHashMap;

use crate::array::PyArray;
use crate::cold;
use crate::convert::NpyIndex;
use crate::dtype::Element;
use crate::error::{BorrowError, NotContiguousError};
use crate::npyffi::{self, PyArrayObject, NPY_ARRAY_WRITEABLE};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BorrowKey {
    /// exclusive range of lowest and highest address covered by array
    range: (*mut u8, *mut u8),
    /// the data address on which address computations are based
    data_ptr: *mut u8,
    /// the greatest common divisor of the strides of the array
    gcd_strides: isize,
}

impl BorrowKey {
    fn from_array<T, D>(array: &PyArray<T, D>) -> Self
    where
        T: Element,
        D: Dimension,
    {
        let range = data_range(array);

        let data_ptr = array.data() as *mut u8;
        let gcd_strides = gcd_strides(array.strides());

        Self {
            range,
            data_ptr,
            gcd_strides,
        }
    }

    fn conflicts(&self, other: &Self) -> bool {
        debug_assert!(self.range.0 <= self.range.1);
        debug_assert!(other.range.0 <= other.range.1);

        if other.range.0 >= self.range.1 || self.range.0 >= other.range.1 {
            return false;
        }

        // The Diophantine equation which describes whether any integers can combine the data pointers and strides of the two arrays s.t.
        // they yield the same element has a solution if and only if the GCD of all strides divides the difference of the data pointers.
        //
        // That solution could be out of bounds which mean that this is still an over-approximation.
        // It appears sufficient to handle typical cases like the color channels of an image,
        // but fails when slicing an array with a step size that does not divide the dimension along that axis.
        //
        // https://users.rust-lang.org/t/math-for-borrow-checking-numpy-arrays/73303
        let ptr_diff = unsafe { self.data_ptr.offset_from(other.data_ptr).abs() };
        let gcd_strides = gcd(self.gcd_strides, other.gcd_strides);

        if ptr_diff % gcd_strides != 0 {
            return false;
        }

        // By default, a conflict is assumed as it is the safe choice without actually solving the aliasing equation.
        true
    }
}

type BorrowFlagsInner = FxHashMap<*mut u8, FxHashMap<BorrowKey, isize>>;

struct BorrowFlags(UnsafeCell<Option<BorrowFlagsInner>>);

unsafe impl Sync for BorrowFlags {}

impl BorrowFlags {
    const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut BorrowFlagsInner {
        (*self.0.get()).get_or_insert_with(Default::default)
    }

    fn acquire(&self, _py: Python, address: *mut u8, key: BorrowKey) -> Result<(), BorrowError> {
        // SAFETY: Having `_py` implies holding the GIL and
        // we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { self.get() };

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(readers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*readers, 0);

                    let new_readers = readers.wrapping_add(1);

                    if new_readers <= 0 {
                        cold();
                        return Err(BorrowError::AlreadyBorrowed);
                    }

                    *readers = new_readers;
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, readers)| key.conflicts(other) && *readers < 0)
                    {
                        cold();
                        return Err(BorrowError::AlreadyBorrowed);
                    }

                    same_base_arrays.insert(key, 1);
                }
            }
            Entry::Vacant(entry) => {
                let mut same_base_arrays =
                    FxHashMap::with_capacity_and_hasher(1, Default::default());
                same_base_arrays.insert(key, 1);
                entry.insert(same_base_arrays);
            }
        }

        Ok(())
    }

    fn release(&self, _py: Python, address: *mut u8, key: BorrowKey) {
        // SAFETY: Having `_py` implies holding the GIL and
        // we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { self.get() };

        let same_base_arrays = borrow_flags.get_mut(&address).unwrap();

        let readers = same_base_arrays.get_mut(&key).unwrap();

        *readers -= 1;

        if *readers == 0 {
            if same_base_arrays.len() > 1 {
                same_base_arrays.remove(&key).unwrap();
            } else {
                borrow_flags.remove(&address).unwrap();
            }
        }
    }

    fn acquire_mut(
        &self,
        _py: Python,
        address: *mut u8,
        key: BorrowKey,
    ) -> Result<(), BorrowError> {
        // SAFETY: Having `_py` implies holding the GIL and
        // we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { self.get() };

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(writers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*writers, 0);

                    cold();
                    return Err(BorrowError::AlreadyBorrowed);
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, writers)| key.conflicts(other) && *writers != 0)
                    {
                        cold();
                        return Err(BorrowError::AlreadyBorrowed);
                    }

                    same_base_arrays.insert(key, -1);
                }
            }
            Entry::Vacant(entry) => {
                let mut same_base_arrays =
                    FxHashMap::with_capacity_and_hasher(1, Default::default());
                same_base_arrays.insert(key, -1);
                entry.insert(same_base_arrays);
            }
        }

        Ok(())
    }

    fn release_mut(&self, _py: Python, address: *mut u8, key: BorrowKey) {
        // SAFETY: Having `_py` implies holding the GIL and
        // we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { self.get() };

        let same_base_arrays = borrow_flags.get_mut(&address).unwrap();

        if same_base_arrays.len() > 1 {
            same_base_arrays.remove(&key).unwrap();
        } else {
            borrow_flags.remove(&address);
        }
    }
}

static BORROW_FLAGS: BorrowFlags = BorrowFlags::new();

/// Read-only borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadwriteArray`],
/// i.e. that only shared references into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
#[repr(C)]
pub struct PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    array: &'py PyArray<T, D>,
    address: *mut u8,
    key: BorrowKey,
}

/// Read-only borrow of a one-dimensional array.
pub type PyReadonlyArray1<'py, T> = PyReadonlyArray<'py, T, Ix1>;

/// Read-only borrow of a two-dimensional array.
pub type PyReadonlyArray2<'py, T> = PyReadonlyArray<'py, T, Ix2>;

/// Read-only borrow of a three-dimensional array.
pub type PyReadonlyArray3<'py, T> = PyReadonlyArray<'py, T, Ix3>;

/// Read-only borrow of a four-dimensional array.
pub type PyReadonlyArray4<'py, T> = PyReadonlyArray<'py, T, Ix4>;

/// Read-only borrow of a five-dimensional array.
pub type PyReadonlyArray5<'py, T> = PyReadonlyArray<'py, T, Ix5>;

/// Read-only borrow of a six-dimensional array.
pub type PyReadonlyArray6<'py, T> = PyReadonlyArray<'py, T, Ix6>;

/// Read-only borrow of an array whose dimensionality is determined at runtime.
pub type PyReadonlyArrayDyn<'py, T> = PyReadonlyArray<'py, T, IxDyn>;

impl<'py, T, D> Deref for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    type Target = PyArray<T, D>;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for PyReadonlyArray<'py, T, D> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        let array: &'py PyArray<T, D> = obj.extract()?;
        Ok(array.readonly())
    }
}

impl<'py, T, D> PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    pub(crate) fn try_new(array: &'py PyArray<T, D>) -> Result<Self, BorrowError> {
        let address = base_address(array);
        let key = BorrowKey::from_array(array);

        BORROW_FLAGS.acquire(array.py(), address, key)?;

        Ok(Self {
            array,
            address,
            key,
        })
    }

    /// Provides an immutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array(&self) -> ArrayView<T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_array() }
    }

    /// Provide an immutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_slice() }
    }

    /// Provide an immutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<&T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.array.get(index) }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N, D> PyReadonlyArray<'py, N, D>
where
    N: nalgebra::Scalar + Element,
    D: Dimension,
{
    /// Try to convert this array into a [`nalgebra::MatrixSlice`] using the given shape and strides.
    #[doc(alias = "nalgebra")]
    pub fn try_as_matrix<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixSlice<N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        unsafe { self.array.try_as_matrix() }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadonlyArray<'py, N, Ix1>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this one-dimensional array into a [`nalgebra::DMatrixSlice`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix(&self) -> nalgebra::DMatrixSlice<N, nalgebra::Dynamic, nalgebra::Dynamic> {
        self.try_as_matrix().unwrap()
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadonlyArray<'py, N, Ix2>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this two-dimensional array into a [`nalgebra::DMatrixSlice`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix(&self) -> nalgebra::DMatrixSlice<N, nalgebra::Dynamic, nalgebra::Dynamic> {
        self.try_as_matrix().unwrap()
    }
}

impl<'a, T, D> Clone for PyReadonlyArray<'a, T, D>
where
    T: Element,
    D: Dimension,
{
    fn clone(&self) -> Self {
        BORROW_FLAGS
            .acquire(self.array.py(), self.address, self.key)
            .unwrap();

        Self {
            array: self.array,
            address: self.address,
            key: self.key,
        }
    }
}

impl<'a, T, D> Drop for PyReadonlyArray<'a, T, D>
where
    T: Element,
    D: Dimension,
{
    fn drop(&mut self) {
        BORROW_FLAGS.release(self.array.py(), self.address, self.key);
    }
}

impl<'py, T, D> fmt::Debug for PyReadonlyArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!(
            "PyReadonlyArray<{}, {}>",
            type_name::<T>(),
            type_name::<D>()
        );

        f.debug_struct(&name).finish()
    }
}

/// Read-write borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadonlyArray`] and no other instances of [`PyReadwriteArray`],
/// i.e. that only a single exclusive reference into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
#[repr(C)]
pub struct PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    array: &'py PyArray<T, D>,
    address: *mut u8,
    key: BorrowKey,
}

/// Read-write borrow of a one-dimensional array.
pub type PyReadwriteArray1<'py, T> = PyReadwriteArray<'py, T, Ix1>;

/// Read-write borrow of a two-dimensional array.
pub type PyReadwriteArray2<'py, T> = PyReadwriteArray<'py, T, Ix2>;

/// Read-write borrow of a three-dimensional array.
pub type PyReadwriteArray3<'py, T> = PyReadwriteArray<'py, T, Ix3>;

/// Read-write borrow of a four-dimensional array.
pub type PyReadwriteArray4<'py, T> = PyReadwriteArray<'py, T, Ix4>;

/// Read-write borrow of a five-dimensional array.
pub type PyReadwriteArray5<'py, T> = PyReadwriteArray<'py, T, Ix5>;

/// Read-write borrow of a six-dimensional array.
pub type PyReadwriteArray6<'py, T> = PyReadwriteArray<'py, T, Ix6>;

/// Read-write borrow of an array whose dimensionality is determined at runtime.
pub type PyReadwriteArrayDyn<'py, T> = PyReadwriteArray<'py, T, IxDyn>;

impl<'py, T, D> Deref for PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    type Target = PyReadonlyArray<'py, T, D>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Exclusive references decay implictly into shared references.
        unsafe { &*(self as *const Self as *const Self::Target) }
    }
}

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for PyReadwriteArray<'py, T, D> {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        let array: &'py PyArray<T, D> = obj.extract()?;
        Ok(array.readwrite())
    }
}

impl<'py, T, D> PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    pub(crate) fn try_new(array: &'py PyArray<T, D>) -> Result<Self, BorrowError> {
        if !array.check_flags(NPY_ARRAY_WRITEABLE) {
            return Err(BorrowError::NotWriteable);
        }

        let address = base_address(array);
        let key = BorrowKey::from_array(array);

        BORROW_FLAGS.acquire_mut(array.py(), address, key)?;

        Ok(Self {
            array,
            address,
            key,
        })
    }

    /// Provides a mutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> ArrayViewMut<T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_array_mut() }
    }

    /// Provide a mutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> Result<&mut [T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.array.as_slice_mut() }
    }

    /// Provide a mutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.array.get_mut(index) }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N, D> PyReadwriteArray<'py, N, D>
where
    N: nalgebra::Scalar + Element,
    D: Dimension,
{
    /// Try to convert this array into a [`nalgebra::MatrixSliceMut`] using the given shape and strides.
    #[doc(alias = "nalgebra")]
    pub fn try_as_matrix_mut<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixSliceMut<N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        unsafe { self.array.try_as_matrix_mut() }
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadwriteArray<'py, N, Ix1>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this one-dimensional array into a [`nalgebra::DMatrixSliceMut`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix_mut(
        &self,
    ) -> nalgebra::DMatrixSliceMut<N, nalgebra::Dynamic, nalgebra::Dynamic> {
        self.try_as_matrix_mut().unwrap()
    }
}

#[cfg(feature = "nalgebra")]
impl<'py, N> PyReadwriteArray<'py, N, Ix2>
where
    N: nalgebra::Scalar + Element,
{
    /// Convert this two-dimensional array into a [`nalgebra::DMatrixSliceMut`] using dynamic strides.
    ///
    /// # Panics
    ///
    /// Panics if the array has negative strides.
    #[doc(alias = "nalgebra")]
    pub fn as_matrix_mut(
        &self,
    ) -> nalgebra::DMatrixSliceMut<N, nalgebra::Dynamic, nalgebra::Dynamic> {
        self.try_as_matrix_mut().unwrap()
    }
}

impl<'py, T> PyReadwriteArray<'py, T, Ix1>
where
    T: Element,
{
    /// Extends or truncates the dimensions of an array.
    ///
    /// Safe wrapper for [`PyArray::resize`].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 10, 1);
    ///     assert_eq!(pyarray.len(), 10);
    ///
    ///     let pyarray = pyarray.readwrite();
    ///     let pyarray = pyarray.resize(100).unwrap();
    ///     assert_eq!(pyarray.len(), 100);
    /// });
    /// ```
    pub fn resize<ID: IntoDimension>(self, dims: ID) -> PyResult<Self> {
        let array = self.array;

        // SAFETY: Ownership of `self` proves exclusive access to the interior of the array.
        unsafe {
            array.resize(dims)?;
        }

        drop(self);

        Ok(Self::try_new(array).unwrap())
    }
}

impl<'a, T, D> Drop for PyReadwriteArray<'a, T, D>
where
    T: Element,
    D: Dimension,
{
    fn drop(&mut self) {
        BORROW_FLAGS.release_mut(self.array.py(), self.address, self.key);
    }
}

impl<'py, T, D> fmt::Debug for PyReadwriteArray<'py, T, D>
where
    T: Element,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!(
            "PyReadwriteArray<{}, {}>",
            type_name::<T>(),
            type_name::<D>()
        );

        f.debug_struct(&name).finish()
    }
}

fn base_address<T, D>(array: &PyArray<T, D>) -> *mut u8 {
    fn inner(py: Python, mut array: *mut PyArrayObject) -> *mut u8 {
        loop {
            let base = unsafe { (*array).base };

            if base.is_null() {
                return array as *mut u8;
            } else if unsafe { npyffi::PyArray_Check(py, base) } != 0 {
                array = base as *mut PyArrayObject;
            } else {
                return base as *mut u8;
            }
        }
    }

    inner(array.py(), array.as_array_ptr())
}

fn data_range<T, D>(array: &PyArray<T, D>) -> (*mut u8, *mut u8)
where
    T: Element,
    D: Dimension,
{
    fn inner(
        shape: &[usize],
        strides: &[isize],
        itemsize: isize,
        data: *mut u8,
    ) -> (*mut u8, *mut u8) {
        let mut start = 0;
        let mut end = 0;

        if shape.iter().all(|dim| *dim != 0) {
            for (&dim, &stride) in shape.iter().zip(strides) {
                let offset = (dim - 1) as isize * stride;

                if offset >= 0 {
                    end += offset;
                } else {
                    start += offset;
                }
            }

            end += itemsize;
        }

        let start = unsafe { data.offset(start) };
        let end = unsafe { data.offset(end) };

        (start, end)
    }

    inner(
        array.shape(),
        array.strides(),
        size_of::<T>() as isize,
        array.data() as *mut u8,
    )
}

fn gcd_strides(strides: &[isize]) -> isize {
    reduce(strides.iter().copied(), gcd).unwrap_or(1)
}

// FIXME(adamreichold): Use `Iterator::reduce` from std when our MSRV reaches 1.51.
fn reduce<I, F>(mut iter: I, f: F) -> Option<I::Item>
where
    I: Iterator,
    F: FnMut(I::Item, I::Item) -> I::Item,
{
    let first = iter.next()?;
    Some(iter.fold(first, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array;
    use pyo3::{types::IntoPyDict, Python};

    use crate::array::{PyArray1, PyArray2, PyArray3};
    use crate::convert::IntoPyArray;

    #[test]
    fn without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(base.is_null());

            let base_address = base_address(array);
            assert_eq!(base_address, array as *const _ as *mut u8);

            let data_range = data_range(array);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(6) } as *mut u8);
        });
    }

    #[test]
    fn with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(array);
            assert_ne!(base_address, array as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(array);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(6) } as *mut u8);
        });
    }

    #[test]
    fn view_without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", array)].into_py_dict(py);
            let view = py
                .eval("array[:,:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(view as *const _ as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base_address = base_address(view);
            assert_ne!(base_address, view as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(view);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(4) } as *mut u8);
        });
    }

    #[test]
    fn view_with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let locals = [("array", array)].into_py_dict(py);
            let view = py
                .eval("array[:,:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(view as *const _ as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(view);
            assert_ne!(base_address, view as *const _ as *mut u8);
            assert_ne!(base_address, array as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(view);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(4) } as *mut u8);
        });
    }

    #[test]
    fn view_of_view_without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", array)].into_py_dict(py);
            let view1 = py
                .eval("array[:,:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(view1 as *const _ as *mut u8, array as *const _ as *mut u8);

            let locals = [("view1", view1)].into_py_dict(py);
            let view2 = py
                .eval("view1[:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(view2 as *const _ as *mut u8, array as *const _ as *mut u8);
            assert_ne!(view2 as *const _ as *mut u8, view1 as *const _ as *mut u8);

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base_address = base_address(view2);
            assert_ne!(base_address, view2 as *const _ as *mut u8);
            assert_ne!(base_address, view1 as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(view2);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(1) } as *mut u8);
        });
    }

    #[test]
    fn view_of_view_with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let locals = [("array", array)].into_py_dict(py);
            let view1 = py
                .eval("array[:,:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();
            assert_ne!(view1 as *const _ as *mut u8, array as *const _ as *mut u8);

            let locals = [("view1", view1)].into_py_dict(py);
            let view2 = py
                .eval("view1[:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(view2 as *const _ as *mut u8, array as *const _ as *mut u8);
            assert_ne!(view2 as *const _ as *mut u8, view1 as *const _ as *mut u8);

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(view2);
            assert_ne!(base_address, view2 as *const _ as *mut u8);
            assert_ne!(base_address, view1 as *const _ as *mut u8);
            assert_ne!(base_address, array as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(view2);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, unsafe { array.data().add(1) } as *mut u8);
        });
    }

    #[test]
    fn view_with_negative_strides() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let locals = [("array", array)].into_py_dict(py);
            let view = py
                .eval("array[::-1,:,::-1]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap();
            assert_ne!(view as *const _ as *mut u8, array as *const _ as *mut u8);

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as *mut u8, array as *const _ as *mut u8);

            let base_address = base_address(view);
            assert_ne!(base_address, view as *const _ as *mut u8);
            assert_eq!(base_address, base as *mut u8);

            let data_range = data_range(view);
            assert_eq!(view.data(), unsafe { array.data().offset(2) });
            assert_eq!(data_range.0, unsafe { view.data().offset(-2) } as *mut u8);
            assert_eq!(data_range.1, unsafe { view.data().offset(4) } as *mut u8);
        });
    }

    #[test]
    fn array_with_zero_dimensions() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 0, 3), false);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(base.is_null());

            let base_address = base_address(array);
            assert_eq!(base_address, array as *const _ as *mut u8);

            let data_range = data_range(array);
            assert_eq!(data_range.0, array.data() as *mut u8);
            assert_eq!(data_range.1, array.data() as *mut u8);
        });
    }

    #[test]
    fn view_with_non_dividing_strides() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (10, 10), false);
            let locals = [("array", array)].into_py_dict(py);

            let view1 = py
                .eval("array[:,::3]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();

            let key1 = BorrowKey::from_array(view1);

            assert_eq!(view1.strides(), &[80, 24]);
            assert_eq!(key1.gcd_strides, 8);

            let view2 = py
                .eval("array[:,1::3]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();

            let key2 = BorrowKey::from_array(view2);

            assert_eq!(view2.strides(), &[80, 24]);
            assert_eq!(key2.gcd_strides, 8);

            let view3 = py
                .eval("array[:,::2]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();

            let key3 = BorrowKey::from_array(view3);

            assert_eq!(view3.strides(), &[80, 16]);
            assert_eq!(key3.gcd_strides, 16);

            let view4 = py
                .eval("array[:,1::2]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap();

            let key4 = BorrowKey::from_array(view4);

            assert_eq!(view4.strides(), &[80, 16]);
            assert_eq!(key4.gcd_strides, 16);

            assert!(!key3.conflicts(&key4));
            assert!(key1.conflicts(&key3));
            assert!(key2.conflicts(&key4));

            // This is a false conflict where all aliasing indices like (0,7) and (2,0) are out of bounds.
            assert!(key1.conflicts(&key2));
        });
    }

    #[test]
    fn borrow_multiple_arrays() {
        Python::with_gil(|py| {
            let array1 = PyArray::<f64, _>::zeros(py, 10, false);
            let array2 = PyArray::<f64, _>::zeros(py, 10, false);

            let base1 = base_address(array1);
            let base2 = base_address(array2);

            let key1 = BorrowKey::from_array(array1);
            let _exclusive1 = array1.readwrite();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base1];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);
            }

            let key2 = BorrowKey::from_array(array2);
            let _shared2 = array2.readonly();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 2);

                let same_base_arrays = &borrow_flags[&base1];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                let same_base_arrays = &borrow_flags[&base2];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key2];
                assert_eq!(flag, 1);
            }
        });
    }

    #[test]
    fn borrow_multiple_views() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);
            let base = base_address(array);

            let locals = [("array", array)].into_py_dict(py);

            let view1 = py
                .eval("array[:5]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();

            let key1 = BorrowKey::from_array(view1);
            let exclusive1 = view1.readwrite();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);
            }

            let view2 = py
                .eval("array[5:]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();

            let key2 = BorrowKey::from_array(view2);
            let shared2 = view2.readonly();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 2);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                let flag = same_base_arrays[&key2];
                assert_eq!(flag, 1);
            }

            let view3 = py
                .eval("array[5:]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();

            let key3 = BorrowKey::from_array(view3);
            let shared3 = view3.readonly();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 2);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                let flag = same_base_arrays[&key2];
                assert_eq!(flag, 2);

                let flag = same_base_arrays[&key3];
                assert_eq!(flag, 2);
            }

            let view4 = py
                .eval("array[7:]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();

            let key4 = BorrowKey::from_array(view4);
            let shared4 = view4.readonly();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 3);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                let flag = same_base_arrays[&key2];
                assert_eq!(flag, 2);

                let flag = same_base_arrays[&key3];
                assert_eq!(flag, 2);

                let flag = same_base_arrays[&key4];
                assert_eq!(flag, 1);
            }

            drop(shared2);

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 3);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                let flag = same_base_arrays[&key2];
                assert_eq!(flag, 1);

                let flag = same_base_arrays[&key3];
                assert_eq!(flag, 1);

                let flag = same_base_arrays[&key4];
                assert_eq!(flag, 1);
            }

            drop(shared3);

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 2);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);

                assert!(!same_base_arrays.contains_key(&key2));

                assert!(!same_base_arrays.contains_key(&key3));

                let flag = same_base_arrays[&key4];
                assert_eq!(flag, 1);
            }

            drop(exclusive1);

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base];
                assert_eq!(same_base_arrays.len(), 1);

                assert!(!same_base_arrays.contains_key(&key1));

                assert!(!same_base_arrays.contains_key(&key2));

                assert!(!same_base_arrays.contains_key(&key3));

                let flag = same_base_arrays[&key4];
                assert_eq!(flag, 1);
            }

            drop(shared4);

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 0);
            }
        });
    }

    #[test]
    fn test_debug_formatting() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            {
                let shared = array.readonly();

                assert_eq!(
                    format!("{:?}", shared),
                    "PyReadonlyArray<f64, ndarray::dimension::dim::Dim<[usize; 3]>>"
                );
            }

            {
                let exclusive = array.readwrite();

                assert_eq!(
                    format!("{:?}", exclusive),
                    "PyReadwriteArray<f64, ndarray::dimension::dim::Dim<[usize; 3]>>"
                );
            }
        });
    }

    #[test]
    #[should_panic(expected = "AlreadyBorrowed")]
    fn cannot_clone_exclusive_borrow_via_deref() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (3, 2, 1), false);

            let exclusive = array.readwrite();
            let _shared = exclusive.clone();
        });
    }

    #[test]
    fn failed_resize_does_not_double_release() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);

            // The view will make the internal reference check of `PyArray_Resize` fail.
            let locals = [("array", array)].into_py_dict(py);
            let _view = py
                .eval("array[:]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();

            let exclusive = array.readwrite();
            assert!(exclusive.resize(100).is_err());
        });
    }

    #[test]
    fn ineffective_resize_does_not_conflict() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, 10, false);

            let exclusive = array.readwrite();
            assert!(exclusive.resize(10).is_ok());
        });
    }
}

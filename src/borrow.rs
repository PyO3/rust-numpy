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
//! The second example shows that non-overlapping and interleaved views which do not alias
//! are currently not supported due to over-approximating which borrows are in conflict.
//!
//! ```rust
//! # use std::panic::{catch_unwind, AssertUnwindSafe};
//! #
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
//!     // Will fail at runtime even though `view1` and `view2`
//!     // do not overlap as they are based on the same array.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _view1 = view1.readwrite();
//!         let _view2 = view2.readwrite();
//!     }));
//!     assert!(res.is_err());
//!
//!     // Will fail at runtime even though `view3` and `view4`
//!     // interleave as they are based on the same array.
//!     let res = catch_unwind(AssertUnwindSafe(|| {
//!         let _view3 = view3.readwrite();
//!         let _view4 = view4.readwrite();
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
//! Note that the current implementation of this is an over-approximation: It will consider all borrows potentially conflicting
//! if the initial arrays have the same object at the end of their [base object chain][base].
//! For example, creating two views of the same underlying array by slicing will always yield potentially conflicting borrows
//! even if the slice indices are chosen so that the two views do not actually share any elements by splitting the array into
//! non-overlapping parts of by interleaving along one of its axes.
//!
//! This does limit the set of programs that can be written using safe Rust in way similar to rustc itself
//! which ensures that all accepted programs are memory safe but does not necessarily accept all memory safe programs.
//! The plan is to refine this checking to correctly handle more involved cases like non-overlapping and interleaved
//! views into the same array and until then the unsafe method [`PyArray::as_array_mut`] can be used as an escape hatch.
//!
//! [base]: https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.NPY_AO.base
#![deny(missing_docs)]

use std::cell::UnsafeCell;
use std::collections::hash_map::{Entry, HashMap};
use std::ops::Deref;

use ndarray::{ArrayView, ArrayViewMut, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use pyo3::{FromPyObject, PyAny, PyResult};

use crate::array::PyArray;
use crate::cold;
use crate::convert::NpyIndex;
use crate::dtype::Element;
use crate::error::{BorrowError, NotContiguousError};
use crate::npyffi::{self, PyArrayObject, NPY_ARRAY_WRITEABLE};

struct BorrowFlags(UnsafeCell<Option<HashMap<usize, isize>>>);

unsafe impl Sync for BorrowFlags {}

impl BorrowFlags {
    const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut HashMap<usize, isize> {
        (*self.0.get()).get_or_insert_with(HashMap::new)
    }

    fn acquire<T, D>(&self, array: &PyArray<T, D>) -> Result<(), BorrowError> {
        let address = base_address(array);

        // SAFETY: Access to `&PyArray<T, D>` implies holding the GIL
        // and we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { BORROW_FLAGS.get() };

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let readers = entry.into_mut();

                let new_readers = readers.wrapping_add(1);

                if new_readers <= 0 {
                    cold();
                    return Err(BorrowError::AlreadyBorrowed);
                }

                *readers = new_readers;
            }
            Entry::Vacant(entry) => {
                entry.insert(1);
            }
        }

        Ok(())
    }

    fn release<T, D>(&self, array: &PyArray<T, D>) {
        let address = base_address(array);

        // SAFETY: Access to `&PyArray<T, D>` implies holding the GIL
        // and we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { BORROW_FLAGS.get() };

        let readers = borrow_flags.get_mut(&address).unwrap();

        *readers -= 1;

        if *readers == 0 {
            borrow_flags.remove(&address).unwrap();
        }
    }

    fn acquire_mut<T, D>(&self, array: &PyArray<T, D>) -> Result<(), BorrowError> {
        let address = base_address(array);

        // SAFETY: Access to `&PyArray<T, D>` implies holding the GIL
        // and we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { BORROW_FLAGS.get() };

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let writers = entry.into_mut();

                if *writers != 0 {
                    cold();
                    return Err(BorrowError::AlreadyBorrowed);
                }

                *writers = -1;
            }
            Entry::Vacant(entry) => {
                entry.insert(-1);
            }
        }

        Ok(())
    }

    fn release_mut<T, D>(&self, array: &PyArray<T, D>) {
        let address = base_address(array);

        // SAFETY: Access to `&PyArray<T, D>` implies holding the GIL
        // and we are not calling into user code which might re-enter this function.
        let borrow_flags = unsafe { self.get() };

        borrow_flags.remove(&address).unwrap();
    }
}

static BORROW_FLAGS: BorrowFlags = BorrowFlags::new();

/// Read-only borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadwriteArray`],
/// i.e. that only shared references into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
pub struct PyReadonlyArray<'py, T, D>(&'py PyArray<T, D>);

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

impl<'py, T, D> Deref for PyReadonlyArray<'py, T, D> {
    type Target = PyArray<T, D>;

    fn deref(&self) -> &Self::Target {
        self.0
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
        BORROW_FLAGS.acquire(array)?;

        Ok(Self(array))
    }

    /// Provides an immutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array(&self) -> ArrayView<T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.0.as_array() }
    }

    /// Provide an immutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.0.as_slice() }
    }

    /// Provide an immutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<&T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.0.get(index) }
    }
}

impl<'a, T, D> Clone for PyReadonlyArray<'a, T, D>
where
    T: Element,
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self::try_new(self.0).unwrap()
    }
}

impl<'a, T, D> Drop for PyReadonlyArray<'a, T, D> {
    fn drop(&mut self) {
        BORROW_FLAGS.release(self.0);
    }
}

/// Read-write borrow of an array.
///
/// An instance of this type ensures that there are no instances of [`PyReadonlyArray`] and no other instances of [`PyReadwriteArray`],
/// i.e. that only a single exclusive reference into the interior of the array can be created safely.
///
/// See the [module-level documentation](self) for more.
pub struct PyReadwriteArray<'py, T, D>(&'py PyArray<T, D>);

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

impl<'py, T, D> Deref for PyReadwriteArray<'py, T, D> {
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

        BORROW_FLAGS.acquire_mut(array)?;

        Ok(Self(array))
    }

    /// Provides a mutable array view of the interior of the NumPy array.
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> ArrayViewMut<T, D> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.0.as_array_mut() }
    }

    /// Provide a mutable slice view of the interior of the NumPy array if it is contiguous.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> Result<&mut [T], NotContiguousError> {
        // SAFETY: Global borrow flags ensure aliasing discipline.
        unsafe { self.0.as_slice_mut() }
    }

    /// Provide a mutable reference to an element of the NumPy array if the index is within bounds.
    #[inline(always)]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut T>
    where
        I: NpyIndex<Dim = D>,
    {
        unsafe { self.0.get_mut(index) }
    }
}

impl<'py, T> PyReadwriteArray<'py, T, Ix1>
where
    T: Element,
{
    /// Extends or truncates the length of a one-dimensional array.
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
    pub fn resize(self, new_elems: usize) -> PyResult<Self> {
        BORROW_FLAGS.release_mut(self.0);

        // SAFETY: Ownership of `self` proves exclusive access to the interior of the array.
        unsafe {
            self.0.resize(new_elems)?;
        }

        BORROW_FLAGS.acquire_mut(self.0)?;

        Ok(self)
    }
}

impl<'a, T, D> Drop for PyReadwriteArray<'a, T, D> {
    fn drop(&mut self) {
        BORROW_FLAGS.release_mut(self.0);
    }
}

// FIXME(adamreichold): This is a coarse approximation and needs to be refined,
// i.e. borrows of non-overlapping views into the same base should not be considered conflicting.
fn base_address<T, D>(array: &PyArray<T, D>) -> usize {
    let py = array.py();
    let mut array = array.as_array_ptr();

    loop {
        let base = unsafe { (*array).base };

        if base.is_null() {
            return array as usize;
        } else if unsafe { npyffi::PyArray_Check(py, base) } != 0 {
            array = base as *mut PyArrayObject;
        } else {
            return base as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array;
    use pyo3::{types::IntoPyDict, Python};

    use crate::array::{PyArray1, PyArray2};
    use crate::convert::IntoPyArray;

    #[test]
    fn without_base_object() {
        Python::with_gil(|py| {
            let array = PyArray::<f64, _>::zeros(py, (1, 2, 3), false);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(base.is_null());

            let base_address = base_address(array);
            assert_eq!(base_address, array as *const _ as usize);
        });
    }

    #[test]
    fn with_base_object() {
        Python::with_gil(|py| {
            let array = Array::<f64, _>::zeros((1, 2, 3)).into_pyarray(py);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(array);
            assert_ne!(base_address, array as *const _ as usize);
            assert_eq!(base_address, base as usize);
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
            assert_ne!(view as *const _ as usize, array as *const _ as usize);

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base_address = base_address(view);
            assert_ne!(base_address, view as *const _ as usize);
            assert_eq!(base_address, base as usize);
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
            assert_ne!(view as *const _ as usize, array as *const _ as usize);

            let base = unsafe { (*view.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(view);
            assert_ne!(base_address, view as *const _ as usize);
            assert_ne!(base_address, array as *const _ as usize);
            assert_eq!(base_address, base as usize);
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
            assert_ne!(view1 as *const _ as usize, array as *const _ as usize);

            let locals = [("view1", view1)].into_py_dict(py);
            let view2 = py
                .eval("view1[:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(view2 as *const _ as usize, array as *const _ as usize);
            assert_ne!(view2 as *const _ as usize, view1 as *const _ as usize);

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base_address = base_address(view2);
            assert_ne!(base_address, view2 as *const _ as usize);
            assert_ne!(base_address, view1 as *const _ as usize);
            assert_eq!(base_address, base as usize);
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
            assert_ne!(view1 as *const _ as usize, array as *const _ as usize);

            let locals = [("view1", view1)].into_py_dict(py);
            let view2 = py
                .eval("view1[:,0]", None, Some(locals))
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap();
            assert_ne!(view2 as *const _ as usize, array as *const _ as usize);
            assert_ne!(view2 as *const _ as usize, view1 as *const _ as usize);

            let base = unsafe { (*view2.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base = unsafe { (*view1.as_array_ptr()).base };
            assert_eq!(base as usize, array as *const _ as usize);

            let base = unsafe { (*array.as_array_ptr()).base };
            assert!(!base.is_null());

            let base_address = base_address(view2);
            assert_ne!(base_address, view2 as *const _ as usize);
            assert_ne!(base_address, view1 as *const _ as usize);
            assert_ne!(base_address, array as *const _ as usize);
            assert_eq!(base_address, base as usize);
        });
    }
}

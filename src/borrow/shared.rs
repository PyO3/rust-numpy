use std::cell::UnsafeCell;
use std::collections::hash_map::Entry;

use num_integer::gcd;
use pyo3::Python;
use rustc_hash::FxHashMap;

use crate::cold;
use crate::error::BorrowError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BorrowKey {
    /// exclusive range of lowest and highest address covered by array
    pub range: (*mut u8, *mut u8),
    /// the data address on which address computations are based
    pub data_ptr: *mut u8,
    /// the greatest common divisor of the strides of the array
    pub gcd_strides: isize,
}

impl BorrowKey {
    pub fn conflicts(&self, other: &Self) -> bool {
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

pub struct BorrowFlags(UnsafeCell<Option<BorrowFlagsInner>>);

unsafe impl Sync for BorrowFlags {}

impl BorrowFlags {
    const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self) -> &mut BorrowFlagsInner {
        (*self.0.get()).get_or_insert_with(Default::default)
    }

    pub fn acquire(
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

    pub fn release(&self, _py: Python, address: *mut u8, key: BorrowKey) {
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

    pub fn acquire_mut(
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

    pub fn release_mut(&self, _py: Python, address: *mut u8, key: BorrowKey) {
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

pub static BORROW_FLAGS: BorrowFlags = BorrowFlags::new();

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::types::IntoPyDict;

    use crate::array::{PyArray, PyArray1};

    use super::super::{base_address, borrow_key};

    #[test]
    fn borrow_multiple_arrays() {
        Python::with_gil(|py| {
            let array1 = PyArray::<f64, _>::zeros(py, 10, false);
            let array2 = PyArray::<f64, _>::zeros(py, 10, false);

            let base1 = base_address(array1);
            let base2 = base_address(array2);

            let key1 = borrow_key(array1);
            let _exclusive1 = array1.readwrite();

            {
                let borrow_flags = unsafe { BORROW_FLAGS.get() };
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base1];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);
            }

            let key2 = borrow_key(array2);
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

            let key1 = borrow_key(view1);
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

            let key2 = borrow_key(view2);
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

            let key3 = borrow_key(view3);
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

            let key4 = borrow_key(view4);
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
}

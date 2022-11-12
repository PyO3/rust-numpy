use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::ffi::{c_void, CString};
use std::os::raw::c_int;
use std::ptr::null;

use num_integer::gcd;
use pyo3::{exceptions::PyTypeError, types::PyCapsule, PyResult, PyTryInto, Python};
use rustc_hash::FxHashMap;

use crate::array::get_array_module;
use crate::cold;
use crate::error::BorrowError;

/// Defines the shared C API used for borrow checking
///
/// This structure will be placed into a capsule at
/// `numpy.core.multiarray._BORROW_CHECKING_API`.
///
/// All functions exposed here assume the GIL is held
/// while they are called.
///
/// Versions are assumed to be backwards-compatible, i.e.
/// an extension which knows version N will work using
/// any API version M as long as M >= N holds.
///
/// Put differently, the only valid changes are adding
/// fields (data or functions) at the end of the structure.
#[repr(C)]
struct Shared {
    version: u64,
    flags: *mut c_void,
    acquire: unsafe extern "C" fn(
        flags: *mut c_void,
        address: *mut u8,
        range_start: *mut u8,
        range_end: *mut u8,
        data_ptr: *mut u8,
        gcd_strides: isize,
    ) -> c_int,
    acquire_mut: unsafe extern "C" fn(
        flags: *mut c_void,
        address: *mut u8,
        range_start: *mut u8,
        range_end: *mut u8,
        data_ptr: *mut u8,
        gcd_strides: isize,
    ) -> c_int,
    release: unsafe extern "C" fn(
        flags: *mut c_void,
        address: *mut u8,
        range_start: *mut u8,
        range_end: *mut u8,
        data_ptr: *mut u8,
        gcd_strides: isize,
    ),
    release_mut: unsafe extern "C" fn(
        flags: *mut c_void,
        address: *mut u8,
        range_start: *mut u8,
        range_end: *mut u8,
        data_ptr: *mut u8,
        gcd_strides: isize,
    ),
}

unsafe impl Send for Shared {}

// These are the entry points which implement the shared borrow checking API:

unsafe extern "C" fn acquire_shared(
    flags: *mut c_void,
    address: *mut u8,
    range_start: *mut u8,
    range_end: *mut u8,
    data_ptr: *mut u8,
    gcd_strides: isize,
) -> c_int {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let flags = &mut *(flags as *mut BorrowFlags);

    let key = BorrowKey {
        range: (range_start, range_end),
        data_ptr,
        gcd_strides,
    };

    match flags.acquire(address, key) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

unsafe extern "C" fn acquire_mut_shared(
    flags: *mut c_void,
    address: *mut u8,
    range_start: *mut u8,
    range_end: *mut u8,
    data_ptr: *mut u8,
    gcd_strides: isize,
) -> c_int {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let flags = &mut *(flags as *mut BorrowFlags);

    let key = BorrowKey {
        range: (range_start, range_end),
        data_ptr,
        gcd_strides,
    };

    match flags.acquire_mut(address, key) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

unsafe extern "C" fn release_shared(
    flags: *mut c_void,
    address: *mut u8,
    range_start: *mut u8,
    range_end: *mut u8,
    data_ptr: *mut u8,
    gcd_strides: isize,
) {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let flags = &mut *(flags as *mut BorrowFlags);

    let key = BorrowKey {
        range: (range_start, range_end),
        data_ptr,
        gcd_strides,
    };

    flags.release(address, key);
}

unsafe extern "C" fn release_mut_shared(
    flags: *mut c_void,
    address: *mut u8,
    range_start: *mut u8,
    range_end: *mut u8,
    data_ptr: *mut u8,
    gcd_strides: isize,
) {
    // SAFETY: GIL must be held when calling `acquire_shared`.
    let flags = &mut *(flags as *mut BorrowFlags);

    let key = BorrowKey {
        range: (range_start, range_end),
        data_ptr,
        gcd_strides,
    };

    flags.release_mut(address, key);
}

// This global state is a cache used to access the shared borrow checking API from this extension:

struct SharedPtr(Cell<*const Shared>);

unsafe impl Sync for SharedPtr {}

static SHARED: SharedPtr = SharedPtr(Cell::new(null()));

fn get_or_insert_shared<'py>(py: Python<'py>) -> PyResult<&'py Shared> {
    let mut shared = SHARED.0.get();

    if shared.is_null() {
        shared = insert_shared(py)?;
    }

    // SAFETY: We inserted the capsule if it was missing
    // and verified that it contains a compatible version.
    Ok(unsafe { &*shared })
}

// This function will publish this extensions version of the shared borrow checking API
// as a capsule placed at `numpy.core.multiarray._BORROW_CHECKING_API` and
// immediately initialize the cache used access it from this extension.

#[cold]
fn insert_shared(py: Python) -> PyResult<*const Shared> {
    let module = get_array_module(py)?;

    let capsule: &PyCapsule = match module.getattr("_BORROW_CHECKING_API") {
        Ok(capsule) => capsule.try_into()?,
        Err(_err) => {
            let flags = Box::into_raw(Box::new(BorrowFlags::default()));

            let shared = Shared {
                version: 1,
                flags: flags as *mut c_void,
                acquire: acquire_shared,
                acquire_mut: acquire_mut_shared,
                release: release_shared,
                release_mut: release_mut_shared,
            };

            let capsule = PyCapsule::new_with_destructor(
                py,
                shared,
                Some(CString::new("_BORROW_CHECKING_API").unwrap()),
                |shared, _ctx| {
                    // SAFETY: `shared.flags` was initialized using `Box::into_raw`.
                    let _ = unsafe { Box::from_raw(shared.flags as *mut BorrowFlags) };
                },
            )?;
            module.setattr("_BORROW_CHECKING_API", capsule)?;
            capsule
        }
    };

    // SAFETY: All versions of the shared borrow checking API start with a version field.
    let version = unsafe { *(capsule.pointer() as *mut u64) };
    if version < 1 {
        return Err(PyTypeError::new_err(format!(
            "Version {} of borrow checking API is not supported by this version of rust-numpy",
            version
        )));
    }

    let shared = capsule.pointer() as *const Shared;
    SHARED.0.set(shared);
    Ok(shared)
}

// These entry points will be used to access the shared borrow checking API from this extension:

pub fn acquire(py: Python, address: *mut u8, key: BorrowKey) -> Result<(), BorrowError> {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    let rc = unsafe {
        (shared.acquire)(
            shared.flags,
            address,
            key.range.0,
            key.range.1,
            key.data_ptr,
            key.gcd_strides,
        )
    };

    match rc {
        0 => Ok(()),
        _ => Err(BorrowError::AlreadyBorrowed),
    }
}

pub fn acquire_mut(py: Python, address: *mut u8, key: BorrowKey) -> Result<(), BorrowError> {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    let rc = unsafe {
        (shared.acquire_mut)(
            shared.flags,
            address,
            key.range.0,
            key.range.1,
            key.data_ptr,
            key.gcd_strides,
        )
    };

    match rc {
        0 => Ok(()),
        _ => Err(BorrowError::AlreadyBorrowed),
    }
}

pub fn release(py: Python, address: *mut u8, key: BorrowKey) {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    unsafe {
        (shared.release)(
            shared.flags,
            address,
            key.range.0,
            key.range.1,
            key.data_ptr,
            key.gcd_strides,
        );
    }
}

pub fn release_mut(py: Python, address: *mut u8, key: BorrowKey) {
    let shared = get_or_insert_shared(py).expect("Interal borrow checking API error");

    unsafe {
        (shared.release_mut)(
            shared.flags,
            address,
            key.range.0,
            key.range.1,
            key.data_ptr,
            key.gcd_strides,
        );
    }
}

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

#[derive(Default)]
struct BorrowFlags(BorrowFlagsInner);

impl BorrowFlags {
    fn acquire(&mut self, address: *mut u8, key: BorrowKey) -> Result<(), ()> {
        let borrow_flags = &mut self.0;

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(readers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*readers, 0);

                    let new_readers = readers.wrapping_add(1);

                    if new_readers <= 0 {
                        cold();
                        return Err(());
                    }

                    *readers = new_readers;
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, readers)| key.conflicts(other) && *readers < 0)
                    {
                        cold();
                        return Err(());
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

    fn release(&mut self, address: *mut u8, key: BorrowKey) {
        let borrow_flags = &mut self.0;

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

    fn acquire_mut(&mut self, address: *mut u8, key: BorrowKey) -> Result<(), ()> {
        let borrow_flags = &mut self.0;

        match borrow_flags.entry(address) {
            Entry::Occupied(entry) => {
                let same_base_arrays = entry.into_mut();

                if let Some(writers) = same_base_arrays.get_mut(&key) {
                    // Zero flags are removed during release.
                    assert_ne!(*writers, 0);

                    cold();
                    return Err(());
                } else {
                    if same_base_arrays
                        .iter()
                        .any(|(other, writers)| key.conflicts(other) && *writers != 0)
                    {
                        cold();
                        return Err(());
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

    fn release_mut(&mut self, address: *mut u8, key: BorrowKey) {
        let borrow_flags = &mut self.0;

        let same_base_arrays = borrow_flags.get_mut(&address).unwrap();

        if same_base_arrays.len() > 1 {
            same_base_arrays.remove(&key).unwrap();
        } else {
            borrow_flags.remove(&address);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::types::IntoPyDict;

    use crate::array::{PyArray, PyArray1};

    use super::super::{base_address, borrow_key};

    fn get_borrow_flags<'py>(py: Python) -> &'py BorrowFlagsInner {
        let shared = get_or_insert_shared(py).unwrap();
        assert_eq!(shared.version, 1);
        unsafe { &(*(shared.flags as *mut BorrowFlags)).0 }
    }

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
                let borrow_flags = get_borrow_flags(py);
                assert_eq!(borrow_flags.len(), 1);

                let same_base_arrays = &borrow_flags[&base1];
                assert_eq!(same_base_arrays.len(), 1);

                let flag = same_base_arrays[&key1];
                assert_eq!(flag, -1);
            }

            let key2 = borrow_key(array2);
            let _shared2 = array2.readonly();

            {
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
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
                let borrow_flags = get_borrow_flags(py);
                assert_eq!(borrow_flags.len(), 0);
            }
        });
    }
}

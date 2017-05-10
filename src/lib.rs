
extern crate python3_sys as pyffi;
extern crate cpython;
extern crate libc;

pub mod array;
pub mod npyffi;

pub use array::PyArray;
pub use npyffi::{PyArrayModule, npy_intp, NPY_TYPES, NPY_ORDER};

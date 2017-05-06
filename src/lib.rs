
extern crate python3_sys as pyffi;
extern crate cpython;
extern crate libc;
extern crate num_complex;

pub mod types;
pub mod array;
pub mod npyffi;

pub use array::PyArray;
pub use npyffi::{PyArrayModule, PyUFuncModule, npy_intp, NPY_TYPES, NPY_ORDER};

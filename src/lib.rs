
extern crate python3_sys as pyffi;
extern crate cpython;
extern crate libc;
extern crate num_complex;
extern crate ndarray;

pub mod types;
pub mod array;
pub mod npyffi;
pub mod error;

pub use array::PyArray;
pub use npyffi::{PyArrayModule, PyUFuncModule, npy_intp, NPY_TYPES, NPY_ORDER};

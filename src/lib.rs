
extern crate python3_sys as pyffi;
extern crate numpy_sys as npffi;
extern crate cpython;

pub mod array;
pub mod descr;

pub use npffi::NPY_TYPES;

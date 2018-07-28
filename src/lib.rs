#![feature(specialization)]

extern crate libc;
extern crate ndarray;
extern crate num_complex;
#[cfg(not(feature = "pyo3-reexport"))]
#[macro_use]
extern crate pyo3;

#[cfg(feature = "pyo3-reexport")]
#[macro_use]
pub extern crate pyo3;

pub mod array;
pub mod convert;
pub mod error;
pub mod npyffi;
pub mod types;

pub use array::PyArray;
pub use convert::{IntoPyArray, ToPyArray};
pub use error::*;
pub use npyffi::{PyArrayModule, PyUFuncModule};
pub use types::*;

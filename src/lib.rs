#![feature(specialization)]

#[macro_use]
extern crate cfg_if;
extern crate libc;
extern crate ndarray;
extern crate num_complex;
#[macro_use]
pub extern crate pyo3;

pub mod array;
pub mod convert;
pub mod error;
pub mod npyffi;
pub mod types;

pub use array::PyArray;
pub use convert::IntoPyArray;
pub use error::*;
pub use npyffi::{PyArrayModule, PyUFuncModule};
pub use types::*;

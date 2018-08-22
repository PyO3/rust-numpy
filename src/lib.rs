//! `rust-numpy` provides Rust interfaces for [NumPy C APIs](https://docs.scipy.org/doc/numpy/reference/c-api.html),
//! especially for [ndarray](https://www.numpy.org/devdocs/reference/arrays.ndarray.html) class.
//!
//! It uses [pyo3](https://github.com/PyO3/pyo3) for rust bindings to cpython, and uses
//! [ndarray](https://github.com/bluss/ndarray) for rust side matrix library.
//!
//! For numpy dependency, it calls `import numpy.core` internally. So you just need numpy
//! installed by `pip install numpy` or other ways in your python environment.
//! You can use both system environment and `virtualenv`.
//!
//! # Example
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! extern crate numpy;
//! extern crate pyo3;
//! use pyo3::prelude::*;
//! use numpy::*;
//! fn main() {
//!     let gil = Python::acquire_gil();
//!     let py = gil.python();
//!     let np = PyArrayModule::import(py).unwrap();
//!     let py_array = array![[1i64, 2], [3, 4]].into_pyarray(py, &np);
//!     assert_eq!(
//!         py_array.as_array().unwrap(),
//!         array![[1i64, 2], [3, 4]].into_dyn(),
//!     );
//! }
//! ```

#![feature(specialization)]

#[macro_use]
extern crate cfg_if;
extern crate libc;
extern crate ndarray;
extern crate num_complex;
#[macro_use]
extern crate pyo3;

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

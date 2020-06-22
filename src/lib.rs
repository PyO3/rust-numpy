//! `rust-numpy` provides Rust interfaces for [NumPy C APIs](https://numpy.org/doc/stable/reference/c-api),
//! especially for [ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html) class.
//!
//! It uses [pyo3](https://github.com/PyO3/pyo3) for rust bindings to cpython, and uses
//! [ndarray](https://github.com/bluss/ndarray) for rust side matrix library.
//!
//! For numpy dependency, it calls `import numpy.core` internally. So you just need numpy
//! installed by `pip install numpy` or other ways in your python environment.
//! You can use both system environment and `virtualenv`.
//!
//! This library loads numpy module automatically. So if numpy is not installed, it simply panics,
//! instead of returing a result.
//!
//! # Example
//!
//! ```
//! #[macro_use]
//! extern crate ndarray;
//! use pyo3::prelude::Python;
//! use numpy::{ToPyArray, PyArray};
//! fn main() {
//!     let gil = Python::acquire_gil();
//!     let py = gil.python();
//!     let py_array = array![[1i64, 2], [3, 4]].to_pyarray(py);
//!     assert_eq!(
//!         py_array.as_array(),
//!         array![[1i64, 2], [3, 4]]
//!     );
//! }
//! ```

#[macro_use]
extern crate cfg_if;
#[macro_use]
extern crate pyo3;

pub mod array;
pub mod convert;
pub mod error;
pub mod npyffi;
mod slice_box;
pub mod types;

pub use crate::array::{
    get_array_module, PyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyArray6,
    PyArrayDyn,
};
pub use crate::convert::{IntoPyArray, NpyIndex, ToNpyDims, ToPyArray};
pub use crate::error::{ErrorKind, IntoPyErr, IntoPyResult};
pub use crate::npyffi::{PY_ARRAY_API, PY_UFUNC_API};
pub use crate::types::{c32, c64, NpyDataType, TypeNum};
pub use ndarray::{Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Test readme
#[doc(hidden)]
pub mod doc_test {
    macro_rules! doc_comment {
        ($x:expr, $($tt:tt)*) => {
            #[doc = $x]
            $($tt)*
        };
    }
    macro_rules! doctest {
        ($x: literal, $y:ident) => {
            doc_comment!(include_str!($x), mod $y {});
        };
    }
    doctest!("../README.md", readme_md);
}

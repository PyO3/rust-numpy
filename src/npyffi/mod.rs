//! Low-Level bindings for NumPy C API.
//!
//! https://docs.scipy.org/doc/numpy/reference/c-api.html
//!
//! Most of functions in this submodule are unsafe.
//! If you use functions in this submodule, you need to understand
//! basic usage of Python C API, especially for the reference counting.
//!
//! - http://docs.python.jp/3/c-api/
//! - http://dgrunwald.github.io/rust-cpython/doc/cpython/

pub mod types;
pub mod objects;
pub mod array;
pub mod ufunc;

pub use self::types::*;
pub use self::objects::*;
pub use self::array::*;
pub use self::ufunc::*;

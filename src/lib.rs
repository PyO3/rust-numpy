#![allow(clippy::missing_safety_doc)] // FIXME

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
//! use numpy::{ToPyArray, PyArray};
//! fn main() {
//!     pyo3::Python::with_gil(|py| {
//!         let py_array = array![[1i64, 2], [3, 4]].to_pyarray(py);
//!         assert_eq!(
//!             py_array.readonly().as_array(),
//!             array![[1i64, 2], [3, 4]]
//!         );
//!     })
//! }
//! ```

#[macro_use]
extern crate cfg_if;
#[macro_use]
extern crate pyo3;

pub mod array;
pub mod convert;
mod dtype;
mod error;
pub mod npyffi;
pub mod npyiter;
mod readonly;
mod slice_box;
mod sum_products;

pub use crate::array::{
    get_array_module, PyArray, PyArray0, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5,
    PyArray6, PyArrayDyn,
};
pub use crate::convert::{IntoPyArray, NpyIndex, ToNpyDims, ToPyArray};
pub use crate::dtype::{c32, c64, DataType, Element, PyArrayDescr};
pub use crate::error::{FromVecError, NotContiguousError, ShapeError};
pub use crate::npyffi::{PY_ARRAY_API, PY_UFUNC_API};
pub use crate::npyiter::{
    IterMode, NpyIterFlag, NpyMultiIter, NpyMultiIterBuilder, NpySingleIter, NpySingleIterBuilder,
};
pub use crate::readonly::{
    PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyReadonlyArray5, PyReadonlyArray6, PyReadonlyArrayDyn,
};
pub use crate::sum_products::{dot, einsum_impl, inner};
pub use ndarray::{array, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

/// Test readme
#[doc(hidden)]
pub mod doc_test {
    macro_rules! doc_comment {
        ($x: expr, $modname: ident) => {
            #[doc = $x]
            mod $modname {}
        };
    }
    doc_comment!(include_str!("../README.md"), readme);
}

/// Create a [PyArray](./array/struct.PyArray.html) with one, two or three dimensions.
/// This macro is backed by
/// [`ndarray::array`](https://docs.rs/ndarray/latest/ndarray/macro.array.html).
///
/// # Example
/// ```
/// pyo3::Python::with_gil(|py| {
///     let array = numpy::pyarray![py, [1, 2], [3, 4]];
///     assert_eq!(
///         array.readonly().as_array(),
///         ndarray::array![[1, 2], [3, 4]]
///     );
/// });
#[macro_export]
macro_rules! pyarray {
    ($py: ident, $([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$([$([$($x,)*],)*],)*], $py)
    }};
    ($py: ident, $([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$([$($x,)*],)*], $py)
    }};
    ($py: ident, $($x:expr),* $(,)*) => {{
        $crate::IntoPyArray::into_pyarray($crate::array![$($x,)*], $py)
    }};
}

//! Defines error types.

use std::error::Error;
use std::fmt;

use pyo3::{exceptions::PyTypeError, PyErr, PyErrArguments, PyObject, Python, ToPyObject};

use crate::dtype::PyArrayDescr;

macro_rules! impl_pyerr {
    ($err_type:ty) => {
        impl Error for $err_type {}

        impl PyErrArguments for $err_type {
            fn arguments(self, py: Python) -> PyObject {
                self.to_string().to_object(py)
            }
        }

        impl From<$err_type> for PyErr {
            fn from(err: $err_type) -> PyErr {
                PyTypeError::new_err(err)
            }
        }
    };
}

/// Represents that dimensionalities of the given arrays do not match.
#[derive(Debug)]
pub struct DimensionalityError {
    from: usize,
    to: usize,
}

impl DimensionalityError {
    pub(crate) fn new(from: usize, to: usize) -> Self {
        Self { from, to }
    }
}

impl fmt::Display for DimensionalityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "dimensionality mismatch:\n from={}, to={}",
            self.from, self.to
        )
    }
}

impl_pyerr!(DimensionalityError);

/// Represents that types of the given arrays do not match.
#[derive(Debug)]
pub struct TypeError {
    from: String,
    to: String,
}

impl TypeError {
    pub(crate) fn new(from: &PyArrayDescr, to: &PyArrayDescr) -> Self {
        let dtype_to_str = |dtype: &PyArrayDescr| {
            dtype
                .str()
                .map_or_else(|_| "(unknown)".into(), |s| s.to_string_lossy().into_owned())
        };
        Self {
            from: dtype_to_str(from),
            to: dtype_to_str(to),
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "type mismatch:\n from={}, to={}", self.from, self.to)
    }
}

impl_pyerr!(TypeError);

/// Represents that given `Vec` cannot be treated as an array.
#[derive(Debug)]
pub struct FromVecError {
    len: usize,
    exp_len: usize,
}

impl FromVecError {
    pub(crate) fn new(len: usize, exp_len: usize) -> Self {
        Self { len, exp_len }
    }
}

impl fmt::Display for FromVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "invalid length: {}, but expected {}",
            self.len, self.exp_len
        )
    }
}

impl_pyerr!(FromVecError);

/// Represents that the given array is not contiguous.
#[derive(Debug)]
pub struct NotContiguousError;

impl fmt::Display for NotContiguousError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The given array is not contiguous")
    }
}

impl_pyerr!(NotContiguousError);

/// Inidcates why borrowing an array failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum BorrowError {
    AlreadyBorrowed,
    NotWriteable,
}

impl fmt::Display for BorrowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::AlreadyBorrowed => write!(f, "The given array is already borrowed"),
            Self::NotWriteable => write!(f, "The given array is not writeable"),
        }
    }
}

impl_pyerr!(BorrowError);

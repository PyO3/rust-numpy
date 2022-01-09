//! Defines error types.

use std::fmt;

use pyo3::{exceptions as exc, PyErr, PyErrArguments, PyObject, Python, ToPyObject};

use crate::dtype::PyArrayDescr;

macro_rules! impl_pyerr {
    ($err_type: ty) => {
        impl std::error::Error for $err_type {}

        impl PyErrArguments for $err_type {
            fn arguments(self, py: Python) -> PyObject {
                format!("{}", self).to_object(py)
            }
        }

        impl std::convert::From<$err_type> for PyErr {
            fn from(err: $err_type) -> PyErr {
                exc::PyTypeError::new_err(err)
            }
        }
    };
}

/// Represents that dimensionalities of the given arrays don't match.
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
        let Self { from, to } = self;
        write!(f, "dimensionality mismatch:\n from={}, to={}", from, to)
    }
}

impl_pyerr!(DimensionalityError);

/// Represents that types of the given arrays don't match.
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
        let Self { from, to } = self;
        write!(f, "type mismatch:\n from={}, to={}", from, to)
    }
}

impl_pyerr!(TypeError);

/// Represents that given vec cannot be treated as array.
#[derive(Debug)]
pub struct FromVecError {
    len1: usize,
    len2: usize,
}

impl FromVecError {
    pub(crate) fn new(len1: usize, len2: usize) -> Self {
        FromVecError { len1, len2 }
    }
}

impl fmt::Display for FromVecError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let FromVecError { len1, len2 } = self;
        write!(
            f,
            "Invalid lenension as an array\n Expected the same length {}, but found {}",
            len1, len2
        )
    }
}

impl_pyerr!(FromVecError);

/// Represents that the array is not contiguous.
#[derive(Debug)]
pub struct NotContiguousError;

impl fmt::Display for NotContiguousError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The given array is not contiguous",)
    }
}

impl_pyerr!(NotContiguousError);

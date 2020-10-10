//! Defines error types.
use crate::DataType;
use pyo3::{exceptions as exc, PyErr, PyErrArguments, PyObject, Python, ToPyObject};
use std::fmt;

/// Represents a dimension and dtype of numpy array.
///
/// Only for error formatting.
#[derive(Debug)]
pub(crate) struct ArrayDim {
    dim: Option<usize>,
    dtype: Option<DataType>,
}

impl fmt::Display for ArrayDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ArrayDim { dim, dtype } = self;
        match (dim, dtype) {
            (Some(dim), Some(dtype)) => write!(f, "dim={:?}, dtype={:?}", dim, dtype),
            (None, Some(dtype)) => write!(f, "dim=_, dtype={:?}", dtype),
            (Some(dim), None) => write!(f, "dim={:?}, dtype=Unknown", dim),
            (None, None) => write!(f, "dim=_, dtype=Unknown"),
        }
    }
}

/// Represents that shapes of the given arrays don't match.
#[derive(Debug)]
pub struct ShapeError {
    from: ArrayDim,
    to: ArrayDim,
}

impl ShapeError {
    pub(crate) fn new(
        from_dtype: &crate::PyArrayDescr,
        from_dim: usize,
        to_type: DataType,
        to_dim: Option<usize>,
    ) -> Self {
        ShapeError {
            from: ArrayDim {
                dim: Some(from_dim),
                dtype: from_dtype.get_datatype(),
            },
            to: ArrayDim {
                dim: to_dim,
                dtype: Some(to_type),
            },
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ShapeError { from, to } = self;
        write!(f, "Shape Mismatch:\n from=({}), to=({})", from, to)
    }
}

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

impl_pyerr!(ShapeError);

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

//! Defines error types.

use array::PyArray;
use convert::ToNpyDims;
use pyo3::*;
use std::error;
use std::fmt;
use types::{NpyDataType, TypeNum};

pub trait IntoPyErr {
    fn into_pyerr(self, msg: &str) -> PyErr;
}

pub trait IntoPyResult {
    type ValueType;
    fn into_pyresult(self, msg: &str) -> PyResult<Self::ValueType>;
}

impl<T, E: IntoPyErr> IntoPyResult for Result<T, E> {
    type ValueType = T;
    fn into_pyresult(self, msg: &str) -> PyResult<T> {
        self.map_err(|e| e.into_pyerr(msg))
    }
}

/// Represents a shape and format of numpy array.
///
/// Only for error formatting.
#[derive(Debug)]
pub struct ArrayFormat {
    pub dims: Box<[usize]>,
    pub dtype: NpyDataType,
}

impl fmt::Display for ArrayFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dims={:?}, dtype={:?}", self.dims, self.dtype)
    }
}

/// Represents a casting error between rust types and numpy array.
#[derive(Debug)]
pub enum ErrorKind {
    /// Error for casting `PyArray` into `ArrayView` or `ArrayViewMut`
    PyToRust { from: NpyDataType, to: NpyDataType },
    /// Error for casting rust's `Vec` into numpy array.
    FromVec { dim1: usize, dim2: usize },
    /// Error in numpy -> numpy data conversion
    PyToPy(Box<(ArrayFormat, ArrayFormat)>),
}

impl ErrorKind {
    pub(crate) fn to_rust(from: i32, to: NpyDataType) -> Self {
        ErrorKind::PyToRust {
            from: NpyDataType::from_i32(from),
            to,
        }
    }
    pub(crate) fn dtype_cast<T: TypeNum>(from: &PyArray<T>, to: NpyDataType) -> Self {
        let dims = from
            .shape()
            .into_iter()
            .map(|&x| x)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let from = ArrayFormat {
            dims: dims.clone(),
            dtype: T::npy_data_type(),
        };
        let to = ArrayFormat { dims, dtype: to };
        ErrorKind::PyToPy(Box::new((from, to)))
    }
    pub(crate) fn dims_cast<T: TypeNum>(from: &PyArray<T>, to_dim: impl ToNpyDims) -> Self {
        let dims_from = from
            .shape()
            .into_iter()
            .map(|&x| x)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let dims_to = to_dim
            .dims_ref()
            .into_iter()
            .map(|&x| x)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let from = ArrayFormat {
            dims: dims_from,
            dtype: T::npy_data_type(),
        };
        let to = ArrayFormat {
            dims: dims_to,
            dtype: T::npy_data_type(),
        };
        ErrorKind::PyToPy(Box::new((from, to)))
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorKind::PyToRust { from, to } => {
                write!(f, "Cast failed: from={:?}, to={:?}", from, to)
            }
            ErrorKind::FromVec { dim1, dim2 } => write!(
                f,
                "Cast failed: Vec To PyArray: expect all dim {} but {} was found",
                dim1, dim2
            ),
            ErrorKind::PyToPy(e) => write!(
                f,
                "Cast failed: from=ndarray({:?}), to=ndarray(dtype={:?})",
                e.0, e.1,
            ),
        }
    }
}

impl error::Error for ErrorKind {}

impl IntoPyErr for ErrorKind {
    fn into_pyerr(self, msg: &str) -> PyErr {
        match self {
            ErrorKind::PyToRust { .. } | ErrorKind::FromVec { .. } | ErrorKind::PyToPy(_) => {
                PyErr::new::<exc::TypeError, _>(format!("{}, msg: {}", self, msg))
            }
        }
    }
}

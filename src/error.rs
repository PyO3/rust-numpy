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
pub enum ArrayCastError {
    /// Error for casting `PyArray` into `ArrayView` or `ArrayViewMut`
    ToRust { from: NpyDataType, to: NpyDataType },
    /// Error for casting rust's `Vec` into numpy array.
    FromVec,
    /// Error in numpy -> numpy data conversion
    Numpy(Box<(ArrayFormat, ArrayFormat)>),
}

impl ArrayCastError {
    pub(crate) fn to_rust(from: i32, to: NpyDataType) -> Self {
        ArrayCastError::ToRust {
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
        ArrayCastError::Numpy(Box::new((from, to)))
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
        ArrayCastError::Numpy(Box::new((from, to)))
    }
}

impl fmt::Display for ArrayCastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArrayCastError::ToRust { from, to } => {
                write!(f, "Cast failed: from={:?}, to={:?}", from, to)
            }
            ArrayCastError::FromVec => write!(f, "Cast failed: FromVec (maybe invalid dimension)"),
            ArrayCastError::Numpy(e) => write!(
                f,
                "Cast failed: from=ndarray({:?}), to=ndarray(dtype={:?})",
                e.0, e.1,
            ),
        }
    }
}

impl error::Error for ArrayCastError {}

impl IntoPyErr for ArrayCastError {
    fn into_pyerr(self, msg: &str) -> PyErr {
        let msg = match self {
            ArrayCastError::ToRust { from, to } => format!(
                "ArrayCastError::ToRust: from: {:?}, to: {:?}, msg: {}",
                from, to, msg
            ),
            ArrayCastError::FromVec => format!("ArrayCastError::FromVec: {}", msg),
            ArrayCastError::Numpy(e) => format!(
                "ArrayCastError::Numpy: from: {:?}, to: {:?}, msg: {}",
                e.0, e.1, msg
            ),
        };
        PyErr::new::<exc::TypeError, _>(msg)
    }
}

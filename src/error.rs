//! Defines error types.

use crate::array::PyArray;
use crate::convert::ToNpyDims;
use crate::types::{NpyDataType, TypeNum};
use pyo3::{exceptions as exc, PyErr, PyResult};
use std::error;
use std::fmt;

pub trait IntoPyErr: Into<PyErr> {
    fn into_pyerr(self) -> PyErr;
    fn into_pyerr_with<D: fmt::Display>(self, _: impl FnOnce() -> D) -> PyErr;
}

pub trait IntoPyResult {
    type ValueType;
    fn into_pyresult(self) -> PyResult<Self::ValueType>;
    fn into_pyresult_with<D: fmt::Display>(
        self,
        _: impl FnOnce() -> D,
    ) -> PyResult<Self::ValueType>;
}

impl<T, E: IntoPyErr> IntoPyResult for Result<T, E> {
    type ValueType = T;
    fn into_pyresult(self) -> PyResult<Self::ValueType> {
        self.map_err(|e| e.into())
    }
    fn into_pyresult_with<D: fmt::Display>(
        self,
        msg: impl FnOnce() -> D,
    ) -> PyResult<Self::ValueType> {
        self.map_err(|e| e.into_pyerr_with(msg))
    }
}

/// Represents a shape and dtype of numpy array.
///
/// Only for error formatting.
#[derive(Debug)]
pub struct ArrayShape {
    pub dims: Box<[usize]>,
    pub dtype: NpyDataType,
}

impl ArrayShape {
    fn boxed_dims(dims: &[usize]) -> Box<[usize]> {
        dims.into_iter()
            .map(|&x| x)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
    fn from_array<T: TypeNum, D>(array: &PyArray<T, D>) -> Self {
        ArrayShape {
            dims: Self::boxed_dims(array.shape()),
            dtype: T::npy_data_type(),
        }
    }
    fn from_dims<T: TypeNum, D: ToNpyDims>(dims: D) -> Self {
        ArrayShape {
            dims: Self::boxed_dims(dims.slice()),
            dtype: T::npy_data_type(),
        }
    }
}

impl fmt::Display for ArrayShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dims={:?}, dtype={:?}", self.dims, self.dtype)
    }
}

/// Represents a dimension and dtype of numpy array.
///
/// Only for error formatting.
#[derive(Debug)]
pub struct ArrayDim {
    pub dim: Option<usize>,
    pub dtype: NpyDataType,
}

impl fmt::Display for ArrayDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(d) = self.dim {
            write!(f, "dim={:?}, dtype={:?}", d, self.dtype)
        } else {
            write!(f, "dim=_, dtype={:?}", self.dtype)
        }
    }
}

/// Represents a casting error between rust types and numpy array.
#[derive(Debug)]
pub enum ErrorKind {
    /// Error for casting `PyArray` into `ArrayView` or `ArrayViewMut`
    PyToRust { from: ArrayDim, to: ArrayDim },
    /// Error for casting rust's `Vec` into numpy array.
    FromVec { dim1: usize, dim2: usize },
    /// Error in numpy -> numpy data conversion
    PyToPy(Box<(ArrayShape, ArrayShape)>),
    /// The array need to be contiguous to finish the opretion
    NotContiguous,
}

impl ErrorKind {
    pub(crate) fn to_rust(
        from_t: i32,
        from_d: usize,
        to_t: NpyDataType,
        to_d: Option<usize>,
    ) -> Self {
        ErrorKind::PyToRust {
            from: ArrayDim {
                dim: Some(from_d),
                dtype: NpyDataType::from_i32(from_t),
            },
            to: ArrayDim {
                dim: to_d,
                dtype: to_t,
            },
        }
    }
    pub(crate) fn dtype_cast<T: TypeNum, D>(from: &PyArray<T, D>, to: NpyDataType) -> Self {
        let from = ArrayShape::from_array(from);
        let to = ArrayShape {
            dims: from.dims.clone(),
            dtype: to,
        };
        ErrorKind::PyToPy(Box::new((from, to)))
    }
    pub(crate) fn dims_cast<T: TypeNum, D>(from: &PyArray<T, D>, to_dim: impl ToNpyDims) -> Self {
        let from = ArrayShape::from_array(from);
        let to = ArrayShape::from_dims::<T, _>(to_dim);
        ErrorKind::PyToPy(Box::new((from, to)))
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::PyToRust { from, to } => {
                write!(f, "Extraction failed:\n from=({}), to=({})", from, to)
            }
            ErrorKind::FromVec { dim1, dim2 } => write!(
                f,
                "Cast failed: Vec To PyArray:\n expect all dim {} but {} was found",
                dim1, dim2
            ),
            ErrorKind::PyToPy(e) => write!(
                f,
                "Cast failed: from=ndarray({}), to=ndarray(dtype={})",
                e.0, e.1,
            ),
            ErrorKind::NotContiguous => write!(f, "This array is not contiguous!"),
        }
    }
}

impl error::Error for ErrorKind {}

impl From<ErrorKind> for PyErr {
    fn from(err: ErrorKind) -> PyErr {
        PyErr::new::<exc::TypeError, _>(format!("{}", err))
    }
}

impl IntoPyErr for ErrorKind {
    fn into_pyerr(self) -> PyErr {
        Into::into(self)
    }
    fn into_pyerr_with<D: fmt::Display>(self, msg: impl FnOnce() -> D) -> PyErr {
        PyErr::new::<exc::TypeError, _>(format!("{}\n context: {}", self, msg()))
    }
}

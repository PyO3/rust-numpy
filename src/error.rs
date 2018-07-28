//! Define Errors

use pyo3::*;
use std::error;
use std::fmt;
use types::NpyDataTypes;

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

/// Error for casting `PyArray` into `ArrayView` or `ArrayViewMut`
#[derive(Debug)]
pub enum ArrayCastError {
    ToRust {
        from: NpyDataTypes,
        to: NpyDataTypes,
    },
    FromVec,
}

impl ArrayCastError {
    pub fn to_rust(from: i32, to: i32) -> Self {
        ArrayCastError::ToRust {
            from: NpyDataTypes::from_i32(from),
            to: NpyDataTypes::from_i32(to),
        }
    }
}

impl fmt::Display for ArrayCastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArrayCastError::ToRust { from, to } => {
                write!(f, "Cast failed: from={:?}, to={:?}", from, to)
            }
            ArrayCastError::FromVec => write!(f, "Cast failed: FromVec (maybe invalid dimension)"),
        }
    }
}

impl error::Error for ArrayCastError {
    fn description(&self) -> &str {
        match self {
            ArrayCastError::ToRust { .. } => "ArrayCast failed(IntoArray)",
            ArrayCastError::FromVec => "ArrayCast failed(FromVec)",
        }
    }
}

impl IntoPyErr for ArrayCastError {
    fn into_pyerr(self, msg: &str) -> PyErr {
        let msg = match self {
            ArrayCastError::ToRust { from, to } => format!(
                "ArrayCastError::ToRust: from: {:?}, to: {:?}, msg: {}",
                from, to, msg
            ),
            ArrayCastError::FromVec => format!("ArrayCastError::FromVec: {}", msg),
        };
        PyErr::new::<exc::TypeError, _>(msg)
    }
}

//! Define Errors

use pyo3::*;
use std::error;
use std::fmt;

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
    IntoArray {
        test: i32,
        truth: i32,
    },
    FromVec,
}

impl ArrayCastError {
    pub fn into_array(test: i32, truth: i32) -> Self {
        ArrayCastError::IntoArray { test, truth }
    }
}

impl fmt::Display for ArrayCastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArrayCastError::IntoArray { test, truth } =>  {
                write!(f, "Cast failed: from={}, to={}", test, truth)
            }
            ArrayCastError::FromVec => {
                write!(f, "Cast failed: FromVec (maybe invalid dimension)")
            }
        }
    }
}

impl error::Error for ArrayCastError {
    fn description(&self) -> &str {
        match self {
            ArrayCastError::IntoArray {..} => "ArrayCast failed(IntoArray)",
            ArrayCastError::FromVec => "ArrayCast failed(FromVec)",
        }
    }
}

impl IntoPyErr for ArrayCastError {
    fn into_pyerr(self, msg: &str) -> PyErr {
        let msg = match self {
            ArrayCastError::IntoArray {..} => {
                format!("rust_numpy::ArrayCastError::IntoArray: {}", msg)
            }
            ArrayCastError::FromVec => format!("rust_numpy::ArrayCastError::FromVec: {}", msg),
        };
        PyErr::new::<exc::TypeError, _>(msg)
    }
}

//! Define Errors

use pyo3::*;
use std::error;
use std::fmt;

pub trait IntoPyErr {
    fn into_pyerr(self, py: Python, msg: &str) -> PyErr;
}

pub trait IntoPyResult {
    type ValueType;
    fn into_pyresult(self, py: Python, message: &str) -> PyResult<Self::ValueType>;
}

impl<T, E: IntoPyErr> IntoPyResult for Result<T, E> {
    type ValueType = T;
    fn into_pyresult(self, py: Python, msg: &str) -> PyResult<T> {
        self.map_err(|e| e.into_pyerr(py, msg))
    }
}

/// Error for casting `PyArray` into `ArrayView` or `ArrayViewMut`
#[derive(Debug)]
pub struct ArrayCastError {
    test: i32,
    truth: i32,
}

impl ArrayCastError {
    pub fn new(test: i32, truth: i32) -> Self {
        Self {
            test: test,
            truth: truth,
        }
    }
}

impl fmt::Display for ArrayCastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cast failed: from={}, to={}", self.test, self.truth)
    }
}

impl error::Error for ArrayCastError {
    fn description(&self) -> &str {
        "Array cast failed"
    }
}

impl IntoPyErr for ArrayCastError {
    fn into_pyerr(self, py: Python, msg: &str) -> PyErr {
        let msg = format!("rust_numpy::ArrayCastError: {}", msg);
        PyErr::new::<exc::TypeError, _>(py, msg)
    }
}

//! Define Errors

use std::error;
use std::fmt;

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

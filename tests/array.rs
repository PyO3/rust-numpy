
extern crate cpython;
extern crate numpy;

use numpy::*;

#[test]
fn array_new() {
    let py = cpython::Python::acquire_gil();
    let _arr = PyArray::zeros(py.python(),
                              &[4, 4],
                              NPY_TYPES::NPY_DOUBLE,
                              NPY_ORDER::NPY_CORDER);
}

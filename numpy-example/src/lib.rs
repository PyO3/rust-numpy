
#[macro_use]
extern crate cpython;
extern crate numpy;

use numpy::*;
use cpython::{PyResult, Python};

py_module_initializer!(librust2py, initlibrust2py, PyInit_librust2py, |py, m| {
    try!(m.add(py, "__doc__", "This module is implemented in Rust."));
    try!(m.add(py, "get_arr", py_fn!(py, get_arr_py())));
    Ok(())
});

fn get_arr_py(py: Python) -> PyResult<PyArray> {
    let np = MultiArray::import(py)?;
    let arr = PyArray::zeros(py,
                             &np,
                             &[3, 5],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    Ok(arr)
}

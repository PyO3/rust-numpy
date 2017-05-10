
#[macro_use]
extern crate cpython;
extern crate numpy;

use numpy::*;
use cpython::{PyResult, Python};

py_module_initializer!(_rust_ext, init_rust_ext, PyInit__rust_ext, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "get_arr", py_fn!(py, get_arr_py()))?;
    Ok(())
});

fn get_arr_py(py: Python) -> PyResult<PyArray> {
    let np = PyArrayModule::import(py)?;
    let arr = PyArray::zeros(py,
                             &np,
                             &[3, 5],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    Ok(arr)
}

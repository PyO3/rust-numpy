
#[macro_use]
extern crate cpython;
extern crate numpy;
extern crate ndarray;

use numpy::*;
use ndarray::*;
use cpython::{PyResult, Python, PyObject};

py_module_initializer!(_rust_ext, init_rust_ext, PyInit__rust_ext, |py, m| {
    m.add(py, "__doc__", "Rust extension for NumPy")?;
    m.add(py, "get_arr", py_fn!(py, get_arr_py()))?;
    m.add(py,
             "mult_two",
             py_fn!(py, mult_two_py(pyarr: PyArray) -> PyResult<PyObject> {
        let arr = pyarr.as_array_mut().unwrap();
        mult_two(arr);
        Ok(py.None())
    }))?;
    Ok(())
});

fn get_arr_py(py: Python) -> PyResult<PyArray> {
    let np = PyArrayModule::import(py)?;
    let arr = PyArray::zeros::<f64>(py, &np, &[3, 5], NPY_CORDER);
    Ok(arr)
}

fn mult_two(mut x: ArrayViewMutD<f64>) {
    x *= 2.0;
}

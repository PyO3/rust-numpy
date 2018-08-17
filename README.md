rust-numpy
===========
[![Build Status](https://travis-ci.org/rust-numpy/rust-numpy.svg?branch=master)](https://travis-ci.org/rust-numpy/rust-numpy)
[![Build status](https://ci.appveyor.com/api/projects/status/fbvref80965w64fu/branch/master?svg=true)](https://ci.appveyor.com/project/kngwyu/rust-numpy/branch/master)
[![Crate](http://meritbadge.herokuapp.com/numpy)](https://crates.io/crates/numpy)
[![docs.rs](https://docs.rs/numpy/badge.svg)](https://docs.rs/numpy)

Rust binding of NumPy C-API

Dependencies
-------------

- [rust-ndarray](https://github.com/bluss/rust-ndarray)
- [rust-cpython](https://github.com/dgrunwald/rust-cpython)

and more (see [Cargo.toml](Cargo.toml))

Example
---------
Please see [example](example) directory for a complete example

```
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
numpy = "*"
cpython = "*"
ndarray = "*"
```

```rust
#[macro_use]
extern crate cpython;
extern crate numpy;
extern crate ndarray;

use numpy::*;
use ndarray::*;
use cpython::{PyResult, Python, PyObject};

/* Pure rust-ndarray functions */

// immutable example
fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
    a * &x + &y
}

// mutable example (no return)
fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
    x *= a;
}

/* rust-cpython wrappers (to be exposed) */

// wrapper of `axpy`
fn axpy_py(py: Python, a: f64, x: PyArray, y: PyArray) -> PyResult<PyArray> {
    let np = PyArrayModule::import(py)?;
    let x = x.as_array().into_pyresult(py, "x must be f64 array")?;
    let y = y.as_array().into_pyresult(py, "y must be f64 array")?;
    Ok(axpy(a, x, y).into_pyarray(py, &np))
}

// wrapper of `mult`
fn mult_py(py: Python, a: f64, x: PyArray) -> PyResult<PyObject> {
    let x = x.as_array_mut().into_pyresult(py, "x must be f64 array")?;
    mult(a, x);
    Ok(py.None()) // Python function must returns
}

/* Define module "_rust_ext" */
py_module_initializer!(_rust_ext, init_rust_ext, PyInit__rust_ext, |py, m| {
    m.add(py, "__doc__", "Rust extension for NumPy")?;
    m.add(py, "axpy", py_fn!(py, axpy_py(a: f64, x: PyArray, y: PyArray)))?;
    m.add(py, "mult", py_fn!(py, mult_py(a: f64, x: PyArray)))?;
    Ok(())
});
```

Contribution
-------------
This project is in pre-alpha version.
We need your feedback. Don't hesitate to open [issue](https://github.com/termoshtt/rust-numpy/issues)!

Version
--------

- v0.2.1
  - NEW: trait `IntoPyErr`, `IntoPyResult` for error translation

- v0.2.0
  - NEW: traits `IntoPyArray`, `ToPyArray`
  - MOD: Interface of `PyArray` creation functions are changed

- v0.1.1
  - Update documents

- v0.1.0
  - First Release
  - Expose unsafe interfase of Array and UFunc API

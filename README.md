rust-numpy
===========
[![Build Status](http://35.187.150.216/api/badges/termoshtt/rust-numpy/status.svg)](http://35.187.150.216/termoshtt/rust-numpy)
[![Crate](http://meritbadge.herokuapp.com/numpy)](https://crates.io/crates/numpy)
[![docs.rs](https://docs.rs/numpy/badge.svg)](https://docs.rs/numpy)

Rust binding of NumPy C-API

Example
---------
Please see [example](example) directory for a complete example

```rust
#[macro_use]
extern crate cpython;
extern crate numpy;

use numpy::*;
use cpython::{PyResult, Python};

py_module_initializer!(_rust_ext, init_rust_ext, PyInit__rust_ext, |py, m| {
    m.add(py, "__doc__", "Rust extension for NumPy")?;
    m.add(py, "get_arr", py_fn!(py, get_arr_py()))?;
    Ok(())
});

fn get_arr_py(py: Python) -> PyResult<PyArray> {
    let np = PyArrayModule::import(py)?;
    let arr = PyArray::zeros::<f64>(py, &np, &[3, 5], NPY_CORDER);
    Ok(arr)
}
```

Contribution
-------------
This project is in pre-alpha version.
We need your feedback. Don't hesitate to open [issue](https://github.com/termoshtt/rust-numpy/issues)!

Version
--------

- v0.1.1 2017/5/11
  - Update documents

- v0.1.0 2017/5/11
  - First Release
  - Expose unsafe interfase of Array and UFunc API

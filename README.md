rust-numpy
===========
[![Build Status](https://travis-ci.org/rust-numpy/rust-numpy.svg?branch=master)](https://travis-ci.org/rust-numpy/rust-numpy)
[![Build status](https://ci.appveyor.com/api/projects/status/bjaru43c7t1alx2x/branch/master?svg=true)](https://ci.appveyor.com/project/kngwyu/rust-numpy/branch/master)
[![Crate](http://meritbadge.herokuapp.com/numpy)](https://crates.io/crates/numpy)

Rust bindings for the NumPy C-API

## API documentation
- [Latest release (possibly broken)](https://docs.rs/numpy)
- [Current Master](https://rust-numpy.github.io/rust-numpy)


## Requirements
- current nightly rust (see https://github.com/PyO3/pyo3/issues/5 for nightly features, and
https://github.com/PyO3/pyo3/blob/master/build.rs for minimum required version)
- some rust libraries
  - [ndarray](https://github.com/bluss/ndarray) for rust-side matrix library
  - [pyo3](https://github.com/PyO3/pyo3) for cpython binding
  - and more (see [Cargo.toml](Cargo.toml))
- [numpy](http://www.numpy.org/) installed in your python environments (e.g., via `pip install numpy`)

**Note**
Starting from 0.3, rust-numpy migrated from rust-cpython to pyo3.
If you want to use rust-cpython, use version 0.2.1 from crates.io.

## Supported Python version

Currently 3.5, 3.6, 3.7 are supported.


## Python2 Support
Version 0.5.0 is the last version that supports Python2.

If you want to compile this library with Python2, please use 0.5.0 from crates.io.

In addition, you have to add a feature flag in `Cargo.toml` like
``` toml
[dependencies.numpy]
version = "0.5.0"
features = ["python2"]
```
.

You can also automatically specify python version in [setup.py](examples/simple-extension/setup.py),
using [setuptools-rust](https://github.com/PyO3/setuptools-rust).


## Example


### Execute a Python program from Rust and get results

``` toml
[package]
name = "numpy-test"

[dependencies]
pyo3 = "0.9.0-alpha.1"
numpy = "0.7.0"
```

```rust
use numpy::{PyArray1, get_array_module};
use pyo3::prelude::{ObjectProtocol, PyResult, Python};
use pyo3::types::PyDict;

fn main() -> Result<(), ()> {
    let gil = Python::acquire_gil();
    main_(gil.python()).map_err(|e| {
        eprintln!("error! :{:?}", e);
        // we can't display python error type via ::std::fmt::Display
        // so print error here manually
        e.print_and_set_sys_last_vars(gil.python());
    })
}

fn main_<'py>(py: Python<'py>) -> PyResult<()> {
    let np = py.import("numpy")?;
    let dict = PyDict::new(py);
    dict.set_item("np", np)?;
    let pyarray: &PyArray1<i32> = py
        .eval("np.absolute(np.array([-1, -2, -3], dtype='int32'))", Some(&dict), None)?
        .extract()?;
    let slice = pyarray.as_slice()?;
    assert_eq!(slice, &[1, 2, 3]);
    Ok(())
}
```

### Write a Python module in Rust

Please see the [examples](examples) directory for a complete example

```toml
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
numpy = "0.7.0"
ndarray = "0.13"

[dependencies.pyo3]
version = "0.9.0-alpha.1"
features = ["extension-module"]
```

```rust
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

#[pymodule]
fn rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<f64>, y: ArrayViewD<f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(
        py: Python,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> Py<PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py).to_owned()
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut();
        mult(a, x);
        Ok(())
    }

    Ok(())
}
```

## Contribution
We need your feedback.

Don't hesitate to open [issues](https://github.com/rust-numpy/rust-numpy/issues)!

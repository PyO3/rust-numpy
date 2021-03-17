rust-numpy
===========
[![Actions Status](https://github.com/pyo3/rust-numpy/workflows/CI/badge.svg)](https://github.com/pyo3/rust-numpy/actions)
[![Crate](http://meritbadge.herokuapp.com/numpy)](https://crates.io/crates/numpy)
[![minimum rustc 1.41](https://img.shields.io/badge/rustc-1.41+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

Rust bindings for the NumPy C-API

## API documentation
- [Latest release (possibly broken)](https://docs.rs/numpy)
- [Current Master](https://pyo3.github.io/rust-numpy)


## Requirements
- Rust >= 1.41.1
  - Basically, our MSRV follows the one of [PyO3](https://github.com/PyO3/pyo3).
- Python >= 3.6
  - Python 3.5 support is dropped from 0.13
- Some Rust libraries
  - [ndarray](https://github.com/bluss/ndarray) for rust-side matrix library
  - [PyO3](https://github.com/PyO3/pyo3) for cpython binding
  - and more (see [Cargo.toml](Cargo.toml))
- [numpy](http://www.numpy.org/) installed in your python environments (e.g., via `pip install numpy`)
  - We recommend `numpy >= 1.16.0`, though older version may work.

**Note**
Starting from 0.3, rust-numpy migrated from rust-cpython to pyo3.
If you want to use rust-cpython, use version 0.2.1 from crates.io.


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

You can also automatically specify python version in `setup.py`,
using [setuptools-rust](https://github.com/PyO3/setuptools-rust).


## Example


### Execute a Python program from Rust and get results

``` toml
[package]
name = "numpy-test"

[dependencies]
pyo3 = "0.13"
numpy = "0.13"
```

```rust
use numpy::PyArray1;
use pyo3::prelude::{PyResult, Python};
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py);
        let pyarray: &PyArray1<i32> = py
            .eval("np.absolute(np.array([-1, -2, -3], dtype='int32'))", Some(locals), None)?
            .extract()?;
        let readonly = pyarray.readonly();
        let slice = readonly.as_slice()?;
        assert_eq!(slice, &[1, 2, 3]);
        Ok(())
    })
}

```

### Write a Python module in Rust

Please see [simple-extension](https://github.com/PyO3/rust-numpy/tree/main/examples/simple-extension)
directory for the complete example.
Also, we have an example project with [ndarray-linalg](https://github.com/PyO3/rust-numpy/tree/main/examples/linalg).

```toml
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
numpy = "0.13"
ndarray = "0.14"

[dependencies.pyo3]
version = "0.13"
features = ["extension-module"]
```

```rust
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
        Ok(())
    }

    Ok(())
}
```

## Conributing
We welcome [issues](https://github.com/rust-numpy/rust-numpy/issues)
and [pull requests](https://github.com/rust-numpy/rust-numpy/pulls).

PyO3's [contrinbuting.md](https://github.com/PyO3/pyo3/blob/main/Contributing.md)
is a nice guide for starting.
Also, we have a [gitter](https://gitter.im/PyO3/Lobby) channel for communicating.


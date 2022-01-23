rust-numpy
===========
[![Actions Status](https://github.com/PyO3/rust-numpy/workflows/CI/badge.svg)](https://github.com/PyO3/rust-numpy/actions)
[![Crate](https://img.shields.io/crates/v/numpy.svg)](https://crates.io/crates/numpy)
[![Minimum rustc 1.48](https://img.shields.io/badge/rustc-1.48+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Documentation](https://docs.rs/numpy/badge.svg)](https://docs.rs/numpy)
[![codecov](https://codecov.io/gh/PyO3/rust-numpy/branch/main/graph/badge.svg)](https://codecov.io/gh/PyO3/rust-numpy)

Rust bindings for the NumPy C-API.

## API documentation
- [Latest release](https://docs.rs/numpy)
- [Current main](https://pyo3.github.io/rust-numpy)

## Requirements
- Rust >= 1.48.0
  - Basically, our MSRV follows the one of [PyO3](https://github.com/PyO3/pyo3)
- Python >= 3.6
  - Python 3.5 support is dropped from 0.13
- Some Rust libraries
  - [ndarray](https://github.com/rust-ndarray/ndarray) for Rust-side matrix library
  - [PyO3](https://github.com/PyO3/pyo3) for Python bindings
  - And more (see [Cargo.toml](Cargo.toml))
- [numpy](https://numpy.org/) installed in your Python environments (e.g., via `pip install numpy`)
  - We recommend `numpy >= 1.16.0`, though older versions may work

## Example

### Write a Python module in Rust

Please see the [simple-extension](https://github.com/PyO3/rust-numpy/tree/main/examples/simple-extension)
directory for the complete example.

Also, we have an example project with [ndarray-linalg](https://github.com/PyO3/rust-numpy/tree/main/examples/linalg).

```toml
[lib]
name = "rust_ext"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.15", features = ["extension-module"] }
numpy = "0.15"
```

```rust
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
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

### Execute a Python program from Rust and get results

``` toml
[package]
name = "numpy-test"

[dependencies]
pyo3 = { version = "0.15", features = ["auto-initialize"] }
numpy = "0.15"
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

## Dependency on ndarray

This crate uses types from `ndarray` in its public API. `ndarray` is re-exported
in the crate root so that you do not need to specify it as a direct dependency.

Furthermore, this crate is compatible with multiple versions of `ndarray` and therefore depends
on a range of semver-incompatible versions, currently `>= 0.13, < 0.16`. Cargo does not
automatically choose a single version of `ndarray` by itself if you depend directly or indirectly
on anything but that exact range. It can therefore be necessary to manually unify these dependencies.

For example, if you specify the following dependencies

```toml
numpy = "0.15"
ndarray = "0.13"
```

this will currently depend on both version `0.13.1` and `0.15.3` of `ndarray` by default
even though `0.13.1` is within the range `>= 0.13, < 0.16`. To fix this, you can run

```sh
cargo update ---package ndarray:0.15.3 --precise 0.13.1
```

to achieve a single dependency on version `0.13.1` of `ndarray`.

## Contributing

We welcome [issues](https://github.com/PyO3/rust-numpy/issues)
and [pull requests](https://github.com/PyO3/rust-numpy/pulls).

PyO3's [Contributing.md](https://github.com/PyO3/pyo3/blob/main/Contributing.md)
is a nice guide for starting.
Also, we have a [Gitter](https://gitter.im/PyO3/Lobby) channel for communicating.


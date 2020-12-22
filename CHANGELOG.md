# Changelog

- v0.12.2
  - Pin PyO3 minor versions to 0.12
  - Pin ndarray minor versions to 0.13

- v0.12.1
  - Fix compile error in Rust 1.39

- v0.12.0
  - Introduce `NpySingleIter` and `NpyMultiIter`.
  - Introduce `PyArrayDescr`.

- v0.11.0
  - `PyArray::get` is now unsafe.
  - Introduce `PyArray::get_owned` and `PyReadonlyArray::get`.

- v0.10.0
  - Remove `ErrorKind` and introduce some concrete error types.
  - `PyArray::as_slice`, `PyArray::as_slice_mut`, `PyArray::as_array`, and `PyArray::as_array_mut` is now unsafe.
  - Introduce `PyArray::as_cell_slice`, `PyArray::to_vec`, and `PyArray::to_owned_array`.
  - Rename `TypeNum` trait `Element`, and `NpyDataType` `DataType`.
  - Update PyO3 to 0.11

- v0.9.0
  - Update PyO3 to 0.10.0

- v0.8.0
  - Update PyO3 to 0.9.0
  - Fix SliceBox initialization

- v0.7.0
  - Update PyO3 to 0.8

- v0.6.0
  - Update PyO3 to 0.7
  - Drop Python2 support

- v0.5.0
  - Update PyO3 to 0.6

- v0.4.0
  - Duplicate `PyArrayModule` and import Numpy API automatically
  - Fix memory leak of `IntoPyArray` and add `ToPyArray` crate
  - PyArray has dimension as type parameter. Now it looks like `PyArray<T, D>`
  - Use `ndarray::IntoDimension` to specify dimension
  - Python2 support

- v0.3.1, v0.3.2
  - Just update dependencies

- v0.3.0
  - Breaking Change: Migrated to pyo3 from rust-cpython
  - Some api addition
  - [Static type checking with PhantomData](https://github.com/rust-numpy/rust-numpy/pull/41)

- v0.2.1
  - NEW: trait `IntoPyErr`, `IntoPyResult` for error translation

- v0.2.0
  - NEW: traits `IntoPyArray`, `ToPyArray`
  - MOD: Interface of `PyArray` creation functions are changed

- v0.1.1
  - Update documents

- v0.1.0
  - First Release
  - Expose unsafe interface of Array and UFunc API

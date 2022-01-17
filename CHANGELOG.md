# Changelog

- Unreleased
  - Support object arrays ([#216](https://github.com/PyO3/rust-numpy/pull/216))
  - Support borrowing arrays that are part of other Python objects via `PyArray::borrow_from_array` ([#230](https://github.com/PyO3/rust-numpy/pull/216))
  - `PyArray::new` is now `unsafe`, as it produces uninitialized arrays ([#220](https://github.com/PyO3/rust-numpy/pull/220))
  - `PyArray::from_exact_iter` does not unsoundly trust `ExactSizeIterator::len` any more ([#262](https://github.com/PyO3/rust-numpy/pull/262))
  - `PyArray::as_cell_slice` was removed as it unsoundly interacts with `PyReadonlyArray` allowing safe code to violate aliasing rules ([#260](https://github.com/PyO3/rust-numpy/pull/260))
  - `rayon` feature is now removed, and directly specifying the feature via `ndarray` dependency is recommended ([#250](https://github.com/PyO3/rust-numpy/pull/250))
  - Descriptors rework and related changes ([#256](https://github.com/PyO3/rust-numpy/pull/256)):
    - Remove `DataType`
    - Add the top-level `dtype` function for easy access to registered dtypes
    - Add `PyArrayDescr::of`, `PyArrayDescr::into_dtype_ptr` and `PyArrayDescr::is_equiv_to`
    - `Element` trait has been simplified to just `IS_COPY` const and `get_dtype` method
    - `Element` is now implemented for `isize`
    - `c32` and `c64` aliases have been replaced with `Complex32` and `Complex64`
    - `ShapeError` has been split into `TypeError` and `DimensionalityError`
    - `i32`, `i64`, `u32` and `u64` are now guaranteed to map to
      `np.int32`, `np.int64`, `np.uint32` and `np.uint64` respectively
    - Remove `cfg_if` dependency

- v0.15.1
  - Make arrays produced via `IntoPyArray`, i.e. those owning Rust data, writeable ([#235](https://github.com/PyO3/rust-numpy/pull/235))
  - Fix thread-safety in internal API globals ([#222](https://github.com/PyO3/rust-numpy/pull/222))

- v0.15.0
  - [Remove resolver from Cargo.toml](https://github.com/PyO3/rust-numpy/pull/202)
  - [Bump PyO3 to 0.15](https://github.com/PyO3/rust-numpy/pull/212)

- v0.14.1
  - [Fix MSRV support](https://github.com/PyO3/rust-numpy/issues/198)

- v0.14
  - Bump PyO3 to 0.14
  - Fix [conversion bug](https://github.com/PyO3/rust-numpy/pull/194)

- v0.13.2
  - Support ndarray 0.15

- v0.13.1
  - Allow ndarray `>=0.13, < 0.15` to work with Rust 1.41.1.
  - Add inner, dot, and einsum
  - Add PyArray0

- v0.13.0
  - Bump num-complex to 0.3
  - Bump ndarray to 0.14
  - Bump pyo3 to 0.13
  - Drop support for Python 3.5 (as it is now end-of-life).
  - Remove unused `python3` feature

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
  - [Static type checking with PhantomData](https://github.com/PyO3/rust-numpy/pull/41)

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

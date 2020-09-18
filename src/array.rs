//! Safe interface for NumPy ndarray
use crate::npyffi::{self, npy_intp, NPY_ORDER, PY_ARRAY_API};
use ndarray::*;
use num_traits::AsPrimitive;
use pyo3::{
    ffi, prelude::*, type_object, types::PyAny, AsPyPointer, PyDowncastError, PyNativeType,
    PyResult,
};
use std::{cell::Cell, mem, os::raw::c_int, ptr, slice};
use std::{iter::ExactSizeIterator, marker::PhantomData};

use crate::convert::{IntoPyArray, NpyIndex, ToNpyDims, ToPyArray};
use crate::error::{FromVecError, NotContiguousError, ShapeError};
use crate::slice_box::SliceBox;
use crate::types::Element;

/// A safe, static-typed interface for
/// [NumPy ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html).
///
/// # Memory location
///
/// - Case1: Constructed via [`IntoPyArray`](../convert/trait.IntoPyArray.html) or
/// [`from_vec`](#method.from_vec) or [`from_owned_array`](#method.from_owned_vec).
///
/// These methods don't allocate memory and use `Box<[T]>` as a internal buffer.
///
/// Please take care that **you cannot use some destructive methods like `resize`,
/// for this kind of array**.
///
/// - Case2: Constructed via other methods, like [`ToPyArray`](../convert/trait.ToPyArray.html) or
/// [`from_slice`](#method.from_slice) or [`from_array`](#from_array).
///
/// These methods allocate memory in Python's private heap.
///
/// In both cases, **PyArray is managed by Python GC.**
/// So you can neither retrieve it nor deallocate it manually.
///
/// # Reference
/// Like [`new`](#method.new), all constractor methods of `PyArray` returns `&PyArray`.
///
/// This design follows
/// [pyo3's ownership concept](https://pyo3.rs/master/doc/pyo3/index.html#ownership-and-lifetimes).
///
///
/// # Data type and Dimension
/// `PyArray` has 2 type parametes `T` and `D`. `T` represents its data type like
/// [`f32`](https://doc.rust-lang.org/std/primitive.f32.html), and `D` represents its dimension.
///
/// All data types you can use implements [Element](../types/trait.Element.html).
///
/// Dimensions are represented by ndarray's
/// [Dimension](https://docs.rs/ndarray/latest/ndarray/trait.Dimension.html) trait.
///
/// Typically, you can use `Ix1, Ix2, ..` for fixed size arrays, and use `IxDyn` for dynamic
/// dimensioned arrays. They're re-exported from `ndarray` crate.
///
/// You can also use various type aliases we provide, like [`PyArray1`](./type.PyArray1.html)
/// or [`PyArrayDyn`](./type.PyArrayDyn.html).
///
/// To specify concrete dimension like `3×4×5`, you can use types which implements ndarray's
/// [`IntoDimension`](https://docs.rs/ndarray/latest/ndarray/dimension/conversion/trait.IntoDimension.html)
/// trait. Typically, you can use array(e.g. `[3, 4, 5]`) or tuple(e.g. `(3, 4, 5)`) as a dimension.
///
/// # Example
/// ```
/// # #[macro_use] extern crate ndarray;
/// use pyo3::{GILGuard, Python};
/// use numpy::PyArray;
/// use ndarray::Array;
/// let gil = Python::acquire_gil();
/// let pyarray = PyArray::arange(gil.python(), 0., 4., 1.).reshape([2, 2]).unwrap();
/// let array = array![[3., 4.], [5., 6.]];
/// assert_eq!(
///     array.dot(&pyarray.readonly().as_array()),
///     array![[8., 15.], [12., 23.]]
/// );
/// ```
pub struct PyArray<T, D>(PyAny, PhantomData<T>, PhantomData<D>);

/// One-dimensional array.
pub type PyArray1<T> = PyArray<T, Ix1>;
/// Two-dimensional array.
pub type PyArray2<T> = PyArray<T, Ix2>;
/// Three-dimensional array.
pub type PyArray3<T> = PyArray<T, Ix3>;
/// Four-dimensional array.
pub type PyArray4<T> = PyArray<T, Ix4>;
/// Five-dimensional array.
pub type PyArray5<T> = PyArray<T, Ix5>;
/// Six-dimensional array.
pub type PyArray6<T> = PyArray<T, Ix6>;
/// Dynamic-dimensional array.
pub type PyArrayDyn<T> = PyArray<T, IxDyn>;

/// Returns a array module.
pub fn get_array_module(py: Python<'_>) -> PyResult<&PyModule> {
    PyModule::import(py, npyffi::array::MOD_NAME)
}

unsafe impl<T, D> type_object::PyLayout<PyArray<T, D>> for npyffi::PyArrayObject {}
impl<T, D> type_object::PySizedLayout<PyArray<T, D>> for npyffi::PyArrayObject {}

pyobject_native_type_convert!(
    PyArray<T, D>,
    npyffi::PyArrayObject,
    *npyffi::PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
    Some("numpy"),
    npyffi::PyArray_Check,
    T, D
);

pyobject_native_type_named!(PyArray<T, D>, T, D);
pyobject_native_type_fmt!(PyArray<T, D>, T, D);

impl<'a, T, D> std::convert::From<&'a PyArray<T, D>> for &'a PyAny {
    fn from(ob: &'a PyArray<T, D>) -> Self {
        unsafe { &*(ob as *const PyArray<T, D> as *const PyAny) }
    }
}

impl<T, D> IntoPy<PyObject> for PyArray<T, D> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl<'a, T: Element, D: Dimension> FromPyObject<'a> for &'a PyArray<T, D> {
    // here we do type-check three times
    // 1. Checks if the object is PyArray
    // 2. Checks if the data type of the array is T
    // 3. Checks if the dimension is same as D
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let array = unsafe {
            if npyffi::PyArray_Check(ob.as_ptr()) == 0 {
                return Err(PyDowncastError::new(ob, "PyArray<T, D>").into());
            }
            &*(ob as *const PyAny as *const PyArray<T, D>)
        };
        let type_ = unsafe { (*(*array.as_array_ptr()).descr).type_num };
        let dim = array.shape().len();
        if T::is_same_type(type_) && D::NDIM.map(|n| n == dim).unwrap_or(true) {
            Ok(array)
        } else {
            Err(ShapeError::new(type_, dim, T::DATA_TYPE, D::NDIM).into())
        }
    }
}

impl<T, D> PyArray<T, D> {
    /// Gets a raw [`PyArrayObject`](../npyffi/objects/struct.PyArrayObject.html) pointer.
    pub fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.as_ptr() as _
    }

    #[inline(always)]
    fn check_flag(&self, flag: c_int) -> bool {
        unsafe { *self.as_array_ptr() }.flags & flag == flag
    }

    #[inline(always)]
    pub(crate) fn get_flag(&self) -> c_int {
        unsafe { *self.as_array_ptr() }.flags
    }

    /// Returns a temporally unwriteable reference of the array.
    pub fn readonly(&self) -> crate::PyReadonlyArray<T, D> {
        self.into()
    }

    /// Returns `true` if the internal data of the array is C-style contiguous
    /// (default of numpy and ndarray) or Fortran-style contiguous.
    ///
    /// # Example
    /// ```
    /// use pyo3::types::IntoPyDict;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py = gil.python();
    /// let array = numpy::PyArray::arange(py, 0, 1, 10);
    /// assert!(array.is_contiguous());
    /// let locals = [("np", numpy::get_array_module(py).unwrap())].into_py_dict(py);
    /// let not_contiguous: &numpy::PyArray1<f32> = py
    ///     .eval("np.zeros((3, 5))[::2, 4]", Some(locals), None)
    ///     .unwrap()
    ///     .downcast()
    ///     .unwrap();
    /// assert!(!not_contiguous.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        self.check_flag(npyffi::NPY_ARRAY_C_CONTIGUOUS)
            | self.check_flag(npyffi::NPY_ARRAY_F_CONTIGUOUS)
    }

    /// Returns `true` if the internal data of the array is Fortran-style contiguous.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.check_flag(npyffi::NPY_ARRAY_F_CONTIGUOUS)
    }

    /// Returns `true` if the internal data of the array is C-style contiguous.
    pub fn is_c_contiguous(&self) -> bool {
        self.check_flag(npyffi::NPY_ARRAY_C_CONTIGUOUS)
    }

    /// Get `Py<PyArray>` from `&PyArray`, which is the owned wrapper of PyObject.
    ///
    /// You can use this method when you have to avoid lifetime annotation to your function args
    /// or return types, like used with pyo3's `pymethod`.
    ///
    /// # Example
    /// ```
    /// use pyo3::{GILGuard, Python, Py};
    /// use numpy::PyArray1;
    /// fn return_py_array() -> Py<PyArray1<i32>> {
    ///    let gil = Python::acquire_gil();
    ///    let array = PyArray1::zeros(gil.python(), [5], false);
    ///    array.to_owned()
    /// }
    /// let gil = Python::acquire_gil();
    /// let array = return_py_array();
    /// assert_eq!(array.as_ref(gil.python()).readonly().as_slice().unwrap(), &[0, 0, 0, 0, 0]);
    /// ```
    pub fn to_owned(&self) -> Py<Self> {
        unsafe { Py::from_borrowed_ptr(self.py(), self.as_ptr()) }
    }

    /// Constructs `PyArray` from raw python object without incrementing reference counts.
    pub unsafe fn from_owned_ptr(py: Python<'_>, ptr: *mut ffi::PyObject) -> &Self {
        py.from_owned_ptr(ptr)
    }

    /// Constructs PyArray from raw python object and increments reference counts.
    pub unsafe fn from_borrowed_ptr(py: Python<'_>, ptr: *mut ffi::PyObject) -> &Self {
        py.from_borrowed_ptr(ptr)
    }

    /// Returns the number of dimensions in the array.
    ///
    /// Same as [numpy.ndarray.ndim](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html)
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
    /// assert_eq!(arr.ndim(), 3);
    /// ```
    // C API: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
    pub fn ndim(&self) -> usize {
        let ptr = self.as_array_ptr();
        unsafe { (*ptr).nd as usize }
    }

    /// Returns a slice which contains how many bytes you need to jump to the next row.
    ///
    /// Same as [numpy.ndarray.strides](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
    /// # Example
    /// ```
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
    /// assert_eq!(arr.strides(), &[240, 48, 8]);
    /// ```
    // C API: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_STRIDES
    pub fn strides(&self) -> &[isize] {
        let n = self.ndim();
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).strides;
            slice::from_raw_parts(p, n)
        }
    }

    /// Returns a slice which contains dimmensions of the array.
    ///
    /// Same as [numpy.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
    /// # Example
    /// ```
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
    /// assert_eq!(arr.shape(), &[4, 5, 6]);
    /// ```
    // C API: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_DIMS
    pub fn shape(&self) -> &[usize] {
        let n = self.ndim();
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).dimensions as *mut usize;
            slice::from_raw_parts(p, n)
        }
    }

    /// Calcurates the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn strides_usize(&self) -> &[usize] {
        let n = self.ndim();
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).strides;
            slice::from_raw_parts(p as *const _, n)
        }
    }

    /// Returns the pointer to the first element of the inner array.
    pub(crate) unsafe fn data(&self) -> *mut T {
        let ptr = self.as_array_ptr();
        (*ptr).data as *mut _
    }

    pub(crate) unsafe fn copy_ptr(&self, other: *const T, len: usize) {
        ptr::copy_nonoverlapping(other, self.data(), len)
    }
}

impl<T: Element, D: Dimension> PyArray<T, D> {
    /// Same as [shape](#method.shape), but returns `D`
    #[inline(always)]
    pub fn dims(&self) -> D {
        D::from_dimension(&Dim(self.shape())).expect("PyArray::dims different dimension")
    }

    fn ndarray_shape(&self) -> StrideShape<D> {
        let shape: Shape<_> = Dim(self.dims()).into();
        let size = mem::size_of::<T>();
        let mut st = D::from_dimension(&Dim(self.strides_usize()))
            .expect("PyArray::ndarray_shape: dimension mismatching");
        st.slice_mut().iter_mut().for_each(|e| *e /= size);
        shape.strides(st)
    }

    /// Creates a new uninitialized PyArray in python heap.
    ///
    /// If `is_fortran == true`, returns Fortran-order array. Else, returns C-order array.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray3::<i32>::new(gil.python(), [4, 5, 6], false);
    /// assert_eq!(pyarray.shape(), &[4, 5, 6]);
    /// ```
    pub fn new<ID>(py: Python, dims: ID, is_fortran: bool) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let flags = if is_fortran { 1 } else { 0 };
        unsafe { PyArray::new_(py, dims, ptr::null_mut(), flags) }
    }

    pub(crate) unsafe fn new_<ID>(
        py: Python,
        dims: ID,
        strides: *const npy_intp,
        flag: c_int,
    ) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let dims = dims.into_dimension();
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            T::ffi_dtype() as i32,
            strides as *mut _, // strides
            ptr::null_mut(),   // data
            0,                 // itemsize
            flag,              // flag
            ptr::null_mut(),   //obj
        );
        Self::from_owned_ptr(py, ptr)
    }

    pub(crate) unsafe fn from_boxed_slice<'py, ID>(
        py: Python<'py>,
        dims: ID,
        strides: *const npy_intp,
        slice: Box<[T]>,
    ) -> &'py Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let dims = dims.into_dimension();
        let container = SliceBox::new(slice);
        let data_ptr = container.data;
        let cell = pyo3::PyClassInitializer::from(container)
            .create_cell(py)
            .expect("Object creation failed.");
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            T::ffi_dtype() as i32,
            strides as *mut _,          // strides
            data_ptr as _,              // data
            mem::size_of::<T>() as i32, // itemsize
            0,                          // flag
            ptr::null_mut(),            //obj
        );
        PY_ARRAY_API.PyArray_SetBaseObject(ptr as *mut npyffi::PyArrayObject, cell as _);
        Self::from_owned_ptr(py, ptr)
    }

    /// Construct a new nd-dimensional array filled with 0.
    ///
    /// If `is_fortran` is true, then
    /// a fortran order array is created, otherwise a C-order array is created.
    ///
    /// See also [PyArray_Zeros](https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Zeros)
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray2;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray: &PyArray2<usize> = PyArray2::zeros(gil.python(), [2, 2], false);
    /// assert_eq!(pyarray.readonly().as_array(), array![[0, 0], [0, 0]]);
    /// ```
    pub fn zeros<ID>(py: Python, dims: ID, is_fortran: bool) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let dims = dims.into_dimension();
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(T::ffi_dtype() as i32);
            let ptr = PY_ARRAY_API.PyArray_Zeros(
                dims.ndim_cint(),
                dims.as_dims_ptr(),
                descr,
                if is_fortran { -1 } else { 0 },
            );
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// Returns the immutable view of the internal data of `PyArray` as slice.
    ///
    /// Please consider the use of safe alternatives
    /// ([`PyReadonlyArray::as_slice`](../struct.PyReadonlyArray.html#method.as_slice)
    /// , [`as_cell_slice`](#method.as_cell_slice) or [`to_vec`](#method.to_vec)) instead of this.
    ///
    /// # Safety
    /// If the internal array is not readonly and can be mutated from Python code,
    /// holding the slice might cause undefined behavior.
    pub unsafe fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        if !self.is_contiguous() {
            Err(NotContiguousError)
        } else {
            Ok(slice::from_raw_parts(self.data(), self.len()))
        }
    }

    /// Returns the view of the internal data of `PyArray` as `&[Cell<T>]`.
    pub fn as_cell_slice(&self) -> Result<&[Cell<T>], NotContiguousError> {
        if !self.is_contiguous() {
            Err(NotContiguousError)
        } else {
            Ok(unsafe { slice::from_raw_parts(self.data() as _, self.len()) })
        }
    }

    /// Returns the view of the internal data of `PyArray` as mutable slice.
    ///
    /// # Safety
    /// If another reference to the internal data exists(e.g., `&[T]` or `ArrayView`),
    /// it might cause undefined behavior.
    ///
    /// In such case, please consider the use of [`as_cell_slice`](#method.as_cell_slice),
    pub unsafe fn as_slice_mut(&self) -> Result<&mut [T], NotContiguousError> {
        if !self.is_contiguous() {
            Err(NotContiguousError)
        } else {
            Ok(slice::from_raw_parts_mut(self.data(), self.len()))
        }
    }

    /// Construct PyArray from
    /// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html).
    ///
    /// This method uses internal [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)
    /// of `ndarray::Array` as numpy array.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::from_owned_array(gil.python(), array![[1, 2], [3, 4]]);
    /// assert_eq!(pyarray.readonly().as_array(), array![[1, 2], [3, 4]]);
    /// ```
    pub fn from_owned_array<'py>(py: Python<'py>, arr: Array<T, D>) -> &'py Self {
        IntoPyArray::into_pyarray(arr, py)
    }

    /// Get the immutable reference of the specified element, with checking the passed index is valid.
    ///
    /// Please consider the use of safe alternatives
    /// ([`PyReadonlyArray::get`](../struct.PyReadonlyArray.html#method.get)
    /// or [`get_owned`](#method.get_owned)) instead of this.
    ///
    /// # Safety
    /// If the internal array is not readonly and can be mutated from Python code,
    /// holding the slice might cause undefined behavior.
    #[inline(always)]
    pub unsafe fn get(&self, index: impl NpyIndex<Dim = D>) -> Option<&T> {
        let offset = index.get_checked::<T>(self.shape(), self.strides())?;
        Some(&*self.data().offset(offset))
    }

    /// Get the immutable reference of the specified element, without checking the
    /// passed index is valid.
    ///
    /// See [NpyIndex](../convert/trait.NpyIndex.html) for what types you can use as index.
    ///
    /// Passing an invalid index can cause undefined behavior(mostly SIGSEGV).
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::arange(gil.python(), 0, 16, 1).reshape([2, 2, 4]).unwrap();
    /// assert_eq!(unsafe { *arr.uget([1, 0, 3]) }, 11);
    /// ```
    #[inline(always)]
    pub unsafe fn uget<Idx>(&self, index: Idx) -> &T
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_unchecked::<T>(self.strides());
        &*self.data().offset(offset)
    }

    /// Same as [uget](#method.uget), but returns `&mut T`.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn uget_mut<Idx>(&self, index: Idx) -> &mut T
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_unchecked::<T>(self.strides());
        &mut *(self.data().offset(offset) as *mut _)
    }

    /// Get dynamic dimensioned array from fixed dimension array.
    ///
    /// See [get](#method.get) for usage.
    pub fn to_dyn(&self) -> &PyArray<T, IxDyn> {
        let python = self.py();
        unsafe { PyArray::from_borrowed_ptr(python, self.as_ptr()) }
    }

    /// Get the copy of the specified element in the array.
    ///
    /// See [NpyIndex](../convert/trait.NpyIndex.html) for what types you can use as index.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray2;
    /// use pyo3::types::IntoPyDict;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py = gil.python();
    /// let locals = [("np", numpy::get_array_module(py).unwrap())].into_py_dict(py);
    /// let array: &PyArray2<i64> = py
    ///     .eval("np.array([[0, 1], [2, 3]], dtype='int64')", Some(locals), None)
    ///     .unwrap()
    ///     .downcast()
    ///     .unwrap();
    /// assert_eq!(array.to_vec().unwrap(), vec![0, 1, 2, 3]);
    /// ```
    pub fn get_owned(&self, index: impl NpyIndex<Dim = D>) -> Option<T> {
        unsafe { self.get(index) }.cloned()
    }

    /// Returns the copy of the internal data of `PyArray` to `Vec`.
    ///
    /// Returns `ErrorKind::NotContiguous` if the internal array is not contiguous.
    /// See also [`as_slice`](#method.as_slice)
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray2;
    /// use pyo3::types::IntoPyDict;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py = gil.python();
    /// let locals = [("np", numpy::get_array_module(py).unwrap())].into_py_dict(py);
    /// let array: &PyArray2<i64> = py
    ///     .eval("np.array([[0, 1], [2, 3]], dtype='int64')", Some(locals), None)
    ///     .unwrap()
    ///     .downcast()
    ///     .unwrap();
    /// assert_eq!(array.to_vec().unwrap(), vec![0, 1, 2, 3]);
    /// ```
    pub fn to_vec(&self) -> Result<Vec<T>, NotContiguousError> {
        unsafe { self.as_slice() }.map(ToOwned::to_owned)
    }

    /// Construct PyArray from `ndarray::ArrayBase`.
    ///
    /// This method allocates memory in Python's heap via numpy api, and then copies all elements
    /// of the array there.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::from_array(gil.python(), &array![[1, 2], [3, 4]]);
    /// assert_eq!(pyarray.readonly().as_array(), array![[1, 2], [3, 4]]);
    /// ```
    pub fn from_array<'py, S>(py: Python<'py>, arr: &ArrayBase<S, D>) -> &'py Self
    where
        S: Data<Elem = T>,
    {
        ToPyArray::to_pyarray(arr, py)
    }

    /// Get the immutable view of the internal data of `PyArray`, as
    /// [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
    ///
    /// Please consider the use of safe alternatives
    /// ([`PyReadonlyArray::as_array`](../struct.PyReadonlyArray.html#method.as_array)
    /// or [`to_array`](#method.to_array)) instead of this.
    ///
    /// # Safety
    /// If the internal array is not readonly and can be mutated from Python code,
    /// holding the `ArrayView` might cause undefined behavior.
    pub unsafe fn as_array(&self) -> ArrayView<'_, T, D> {
        ArrayView::from_shape_ptr(self.ndarray_shape(), self.data())
    }

    /// Returns the internal array as `ArrayViewMut`. See also [`as_array`](#method.as_array).
    ///
    /// # Safety
    /// If another reference to the internal data exists(e.g., `&[T]` or `ArrayView`),
    /// it might cause undefined behavior.
    pub unsafe fn as_array_mut(&self) -> ArrayViewMut<'_, T, D> {
        ArrayViewMut::from_shape_ptr(self.ndarray_shape(), self.data())
    }

    /// Get a copy of `PyArray` as
    /// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html).
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py_array = PyArray::arange(gil.python(), 0, 4, 1).reshape([2, 2]).unwrap();
    /// assert_eq!(
    ///     py_array.to_owned_array(),
    ///     array![[0, 1], [2, 3]]
    /// )
    /// ```
    pub fn to_owned_array(&self) -> Array<T, D> {
        unsafe { self.as_array() }.to_owned()
    }
}

impl<T: Element> PyArray<T, Ix1> {
    /// Construct one-dimension PyArray from slice.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let array = [1, 2, 3, 4, 5];
    /// let pyarray = PyArray::from_slice(gil.python(), &array);
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// ```
    pub fn from_slice<'py>(py: Python<'py>, slice: &[T]) -> &'py Self {
        let array = PyArray::new(py, [slice.len()], false);
        unsafe {
            array.copy_ptr(slice.as_ptr(), slice.len());
        }
        array
    }

    /// Construct one-dimension PyArray
    /// from [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html).
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let pyarray = PyArray::from_vec(gil.python(), vec);
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// ```
    pub fn from_vec<'py>(py: Python<'py>, vec: Vec<T>) -> &'py Self {
        IntoPyArray::into_pyarray(vec, py)
    }

    /// Construct one-dimension PyArray from a type which implements
    /// [`ExactSizeIterator`](https://doc.rust-lang.org/std/iter/trait.ExactSizeIterator.html).
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// use std::collections::BTreeSet;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let pyarray = PyArray::from_iter(gil.python(), vec.iter().map(|&x| x));
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// ```
    pub fn from_exact_iter(py: Python<'_>, iter: impl ExactSizeIterator<Item = T>) -> &Self {
        let array = Self::new(py, [iter.len()], false);
        unsafe {
            for (i, item) in iter.enumerate() {
                *array.uget_mut([i]) = item;
            }
        }
        array
    }

    /// Construct one-dimension PyArray from a type which implements
    /// [`IntoIterator`](https://doc.rust-lang.org/std/iter/trait.IntoIterator.html).
    ///
    /// If no reliable [`size_hint`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.size_hint) is available,
    /// this method can allocate memory multiple time, which can hurt performance.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// use std::collections::BTreeSet;
    /// let gil = pyo3::Python::acquire_gil();
    /// let set: BTreeSet<u32> = [4, 3, 2, 5, 1].into_iter().cloned().collect();
    /// let pyarray = PyArray::from_iter(gil.python(), set);
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// ```
    pub fn from_iter(py: Python<'_>, iter: impl IntoIterator<Item = T>) -> &Self {
        let iter = iter.into_iter();
        let (min_len, max_len) = iter.size_hint();
        let mut capacity = max_len.unwrap_or_else(|| min_len.max(512 / mem::size_of::<T>()));
        let array = Self::new(py, [capacity], false);
        let mut length = 0;
        unsafe {
            for (i, item) in iter.enumerate() {
                length += 1;
                if length > capacity {
                    capacity *= 2;
                    array
                        .resize(capacity)
                        .expect("PyArray::from_iter: Failed to allocate memory");
                }
                *array.uget_mut([i]) = item;
            }
        }
        if capacity > length {
            array.resize(length).unwrap()
        }
        array
    }

    /// Extends or trancates the length of 1 dimension PyArray.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::arange(gil.python(), 0, 10, 1);
    /// assert_eq!(pyarray.len(), 10);
    /// pyarray.resize(100).unwrap();
    /// assert_eq!(pyarray.len(), 100);
    /// ```
    pub fn resize(&self, new_elems: usize) -> PyResult<()> {
        self.resize_([new_elems], 1, NPY_ORDER::NPY_ANYORDER)
    }

    fn resize_<D: IntoDimension>(
        &self,
        dims: D,
        check_ref: c_int,
        order: NPY_ORDER,
    ) -> PyResult<()> {
        let dims = dims.into_dimension();
        let mut np_dims = dims.to_npy_dims();
        let res = unsafe {
            PY_ARRAY_API.PyArray_Resize(
                self.as_array_ptr(),
                &mut np_dims as *mut npyffi::PyArray_Dims,
                check_ref,
                order,
            )
        };
        if res.is_null() {
            Err(PyErr::fetch(self.py()))
        } else {
            Ok(())
        }
    }
}

impl<T: Element> PyArray<T, Ix2> {
    /// Construct a two-dimension PyArray from `Vec<Vec<T>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns `ArrayCastError`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec2 = vec![vec![1, 2, 3]; 2];
    /// let pyarray = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    /// assert_eq!(pyarray.readonly().as_array(), array![[1, 2, 3], [1, 2, 3]]);
    /// assert!(PyArray::from_vec2(gil.python(), &[vec![1], vec![2, 3]]).is_err());
    /// ```
    pub fn from_vec2<'py>(py: Python<'py>, v: &[Vec<T>]) -> Result<&'py Self, FromVecError> {
        let last_len = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != last_len) {
            return Err(FromVecError::new(v.len(), last_len));
        }
        let dims = [v.len(), last_len];
        let array = Self::new(py, dims, false);
        unsafe {
            for (y, vy) in v.iter().enumerate() {
                for (x, vyx) in vy.iter().enumerate() {
                    *array.uget_mut([y, x]) = vyx.clone();
                }
            }
        }
        Ok(array)
    }
}

impl<T: Element> PyArray<T, Ix3> {
    /// Construct a three-dimension PyArray from `Vec<Vec<Vec<T>>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns error.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec2 = vec![vec![vec![1, 2]; 2]; 2];
    /// let pyarray = PyArray::from_vec3(gil.python(), &vec2).unwrap();
    /// assert_eq!(
    ///     pyarray.readonly().as_array(),
    ///     array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
    /// );
    /// assert!(PyArray::from_vec3(gil.python(), &[vec![vec![1], vec![]]]).is_err());
    /// ```
    pub fn from_vec3<'py>(py: Python<'py>, v: &[Vec<Vec<T>>]) -> Result<&'py Self, FromVecError> {
        let len2 = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != len2) {
            return Err(FromVecError::new(v.len(), len2));
        }
        let len3 = v.last().map_or(0, |v| v.last().map_or(0, |v| v.len()));
        if v.iter().any(|v| v.iter().any(|v| v.len() != len3)) {
            return Err(FromVecError::new(v.len(), len3));
        }
        let dims = [v.len(), len2, len3];
        let array = Self::new(py, dims, false);
        unsafe {
            for (z, vz) in v.iter().enumerate() {
                for (y, vzy) in vz.iter().enumerate() {
                    for (x, vzyx) in vzy.iter().enumerate() {
                        *array.uget_mut([z, y, x]) = vzyx.clone();
                    }
                }
            }
        }
        Ok(array)
    }
}

impl<T: Element, D> PyArray<T, D> {
    /// Copies self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = PyArray::<i64, _>::new(gil.python(), [3], false);
    /// assert!(pyarray_f.copy_to(pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.readonly().as_slice().unwrap(), &[2, 3, 4]);
    /// ```
    pub fn copy_to<U: Element>(&self, other: &PyArray<U, D>) -> PyResult<()> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_CopyInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(PyErr::fetch(self.py()))
        } else {
            Ok(())
        }
    }

    /// Cast the `PyArray<T>` to `PyArray<U>`, by allocating a new array.
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = pyarray_f.cast::<i32>(false).unwrap();
    /// assert_eq!(pyarray_i.readonly().as_slice().unwrap(), &[2, 3, 4]);
    pub fn cast<'py, U: Element>(&'py self, is_fortran: bool) -> PyResult<&'py PyArray<U, D>> {
        let ptr = unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(U::ffi_dtype() as i32);
            PY_ARRAY_API.PyArray_CastToType(
                self.as_array_ptr(),
                descr,
                if is_fortran { -1 } else { 0 },
            )
        };
        if ptr.is_null() {
            Err(PyErr::fetch(self.py()))
        } else {
            Ok(unsafe { PyArray::<U, D>::from_owned_ptr(self.py(), ptr) })
        }
    }

    /// Construct a new array which has same values as self, same matrix order, but has different
    /// dimensions specified by `dims`.
    ///
    /// Since a returned array can contain a same pointer as self, we highly recommend to drop an
    /// old array, if this method returns `Ok`.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate ndarray;
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let array = PyArray::from_exact_iter(gil.python(), 0..9);
    /// let array = array.reshape([3, 3]).unwrap();
    /// assert_eq!(array.readonly().as_array(), array![[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
    /// assert!(array.reshape([5]).is_err());
    /// ```
    #[inline(always)]
    pub fn reshape<'py, ID, D2>(&'py self, dims: ID) -> PyResult<&'py PyArray<T, D2>>
    where
        ID: IntoDimension<Dim = D2>,
        D2: Dimension,
    {
        self.reshape_with_order(dims, NPY_ORDER::NPY_ANYORDER)
    }

    /// Same as [reshape](method.reshape.html), but you can change the order of returned matrix.
    pub fn reshape_with_order<'py, ID, D2>(
        &'py self,
        dims: ID,
        order: NPY_ORDER,
    ) -> PyResult<&'py PyArray<T, D2>>
    where
        ID: IntoDimension<Dim = D2>,
        D2: Dimension,
    {
        let dims = dims.into_dimension();
        let mut np_dims = dims.to_npy_dims();
        let ptr = unsafe {
            PY_ARRAY_API.PyArray_Newshape(
                self.as_array_ptr(),
                &mut np_dims as *mut npyffi::PyArray_Dims,
                order,
            )
        };
        if ptr.is_null() {
            Err(PyErr::fetch(self.py()))
        } else {
            Ok(unsafe { PyArray::<T, D2>::from_owned_ptr(self.py(), ptr) })
        }
    }
}

impl<T: Element + AsPrimitive<f64>> PyArray<T, Ix1> {
    /// Return evenly spaced values within a given interval.
    /// Same as [numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html).
    ///
    /// See also [PyArray_Arange](https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Arange).
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::arange(gil.python(), 2.0, 4.0, 0.5);
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[2.0, 2.5, 3.0, 3.5]);
    /// let pyarray = PyArray::arange(gil.python(), -2, 4, 3);
    /// assert_eq!(pyarray.readonly().as_slice().unwrap(), &[-2, 1]);
    pub fn arange(py: Python, start: T, stop: T, step: T) -> &Self {
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_Arange(
                start.as_(),
                stop.as_(),
                step.as_(),
                T::ffi_dtype() as i32,
            );
            Self::from_owned_ptr(py, ptr)
        }
    }
}

#[test]
fn test_get_unchecked() {
    let gil = pyo3::Python::acquire_gil();
    let array = PyArray::from_slice(gil.python(), &[1i32, 2, 3]);
    unsafe {
        assert_eq!(*array.uget([1]), 2);
    }
}

#[test]
fn test_dyn_to_owned_array() {
    let gil = pyo3::Python::acquire_gil();
    let array = PyArray::from_vec2(gil.python(), &[vec![1, 2], vec![3, 4]]).unwrap();
    array.to_dyn().to_owned_array();
}

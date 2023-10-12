//! Safe interface for NumPy's [N-dimensional arrays][ndarray]
//!
//! [ndarray]: https://numpy.org/doc/stable/reference/arrays.ndarray.html

use std::{
    marker::PhantomData,
    mem,
    ops::Deref,
    os::raw::{c_int, c_void},
    ptr, slice,
};

use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dim, Dimension, IntoDimension, Ix0, Ix1,
    Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RawArrayView, RawArrayViewMut, RawData, ShapeBuilder,
    StrideShape,
};
use num_traits::AsPrimitive;
use pyo3::{
    ffi, pyobject_native_type_base, types::PyModule, AsPyPointer, FromPyObject, IntoPy, Py, PyAny,
    PyClassInitializer, PyDowncastError, PyErr, PyNativeType, PyObject, PyResult, PyTypeInfo,
    Python, ToPyObject,
};

use crate::borrow::{PyReadonlyArray, PyReadwriteArray};
use crate::cold;
use crate::convert::{ArrayExt, IntoPyArray, NpyIndex, ToNpyDims, ToPyArray};
use crate::dtype::Element;
use crate::error::{
    BorrowError, DimensionalityError, FromVecError, IgnoreError, NotContiguousError, TypeError,
    DIMENSIONALITY_MISMATCH_ERR, MAX_DIMENSIONALITY_ERR,
};
use crate::npyffi::{self, npy_intp, NPY_ORDER, PY_ARRAY_API};
use crate::slice_container::PySliceContainer;
use crate::untyped_array::PyUntypedArray;

/// A safe, statically-typed wrapper for NumPy's [`ndarray`][ndarray] class.
///
/// # Memory location
///
/// - Allocated by Rust: Constructed via [`IntoPyArray`] or
/// [`from_vec`][Self::from_vec] or [`from_owned_array`][Self::from_owned_array].
///
/// These methods transfers ownership of the Rust allocation into a suitable Python object
/// and uses the memory as the internal buffer backing the NumPy array.
///
/// Please note that some destructive methods like [`resize`][Self::resize] will fail
/// when used with this kind of array as NumPy cannot reallocate the internal buffer.
///
/// - Allocated by NumPy: Constructed via other methods, like [`ToPyArray`] or
/// [`from_slice`][Self::from_slice] or [`from_array`][Self::from_array].
///
/// These methods allocate memory in Python's private heap via NumPy's API.
///
/// In both cases, `PyArray` is managed by Python so it can neither be moved from
/// nor deallocated manually.
///
/// # References
///
/// Like [`new`][Self::new], all constructor methods of `PyArray` return a shared reference `&PyArray`
/// instead of an owned value. This design follows [PyO3's ownership concept][pyo3-memory],
/// i.e. the return value is GIL-bound owning reference into Python's heap.
///
/// # Element type and dimensionality
///
/// `PyArray` has two type parametes `T` and `D`.
/// `T` represents the type of its elements, e.g. [`f32`] or [`PyObject`].
/// `D` represents its dimensionality, e.g [`Ix2`][type@Ix2] or [`IxDyn`][type@IxDyn].
///
/// Element types are Rust types which implement the [`Element`] trait.
/// Dimensions are represented by the [`ndarray::Dimension`] trait.
///
/// Typically, `Ix1, Ix2, ...` are used for fixed dimensionality arrays,
/// and `IxDyn` is used for dynamic dimensionality arrays. Type aliases
/// for combining `PyArray` with these types are provided, e.g. [`PyArray1`] or [`PyArrayDyn`].
///
/// To specify concrete dimension like `3×4×5`, types which implement the [`ndarray::IntoDimension`]
/// trait are used. Typically, this means arrays like `[3, 4, 5]` or tuples like `(3, 4, 5)`.
///
/// # Example
///
/// ```
/// use numpy::PyArray;
/// use ndarray::{array, Array};
/// use pyo3::Python;
///
/// Python::with_gil(|py| {
///     let pyarray = PyArray::arange(py, 0., 4., 1.).reshape([2, 2]).unwrap();
///     let array = array![[3., 4.], [5., 6.]];
///
///     assert_eq!(
///         array.dot(&pyarray.readonly().as_array()),
///         array![[8., 15.], [12., 23.]]
///     );
/// });
/// ```
///
/// [ndarray]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
/// [pyo3-memory]: https://pyo3.rs/main/memory.html
#[repr(transparent)]
pub struct PyArray<T, D>(PyAny, PhantomData<T>, PhantomData<D>);

/// Zero-dimensional array.
pub type PyArray0<T> = PyArray<T, Ix0>;
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

/// Returns a handle to NumPy's multiarray module.
pub fn get_array_module<'py>(py: Python<'py>) -> PyResult<&PyModule> {
    PyModule::import(py, npyffi::array::MOD_NAME)
}

unsafe impl<T: Element, D: Dimension> PyTypeInfo for PyArray<T, D> {
    type AsRefTarget = Self;

    const NAME: &'static str = "PyArray<T, D>";
    const MODULE: Option<&'static str> = Some("numpy");

    fn type_object_raw<'py>(py: Python<'py>) -> *mut ffi::PyTypeObject {
        unsafe { npyffi::PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type) }
    }

    fn is_type_of(ob: &PyAny) -> bool {
        Self::extract::<IgnoreError>(ob).is_ok()
    }
}

pyobject_native_type_base!(PyArray<T, D>; T; D);

impl<T, D> AsRef<PyAny> for PyArray<T, D> {
    #[inline]
    fn as_ref(&self) -> &PyAny {
        &self.0
    }
}

impl<T, D> Deref for PyArray<T, D> {
    type Target = PyUntypedArray;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_untyped()
    }
}

unsafe impl<T, D> AsPyPointer for PyArray<T, D> {
    #[inline]
    fn as_ptr(&self) -> *mut ffi::PyObject {
        self.0.as_ptr()
    }
}

impl<T, D> IntoPy<Py<PyArray<T, D>>> for &'_ PyArray<T, D> {
    #[inline]
    fn into_py<'py>(self, py: Python<'py>) -> Py<PyArray<T, D>> {
        unsafe { Py::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl<T, D> From<&'_ PyArray<T, D>> for Py<PyArray<T, D>> {
    #[inline]
    fn from(other: &PyArray<T, D>) -> Self {
        unsafe { Py::from_borrowed_ptr(other.py(), other.as_ptr()) }
    }
}

impl<'a, T, D> From<&'a PyArray<T, D>> for &'a PyAny {
    fn from(ob: &'a PyArray<T, D>) -> Self {
        unsafe { &*(ob as *const PyArray<T, D> as *const PyAny) }
    }
}

impl<T, D> IntoPy<PyObject> for PyArray<T, D> {
    fn into_py<'py>(self, py: Python<'py>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl<'py, T: Element, D: Dimension> FromPyObject<'py> for &'py PyArray<T, D> {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        PyArray::extract(ob)
    }
}

impl<T, D> PyArray<T, D> {
    /// Access an untyped representation of this array.
    #[inline(always)]
    pub fn as_untyped(&self) -> &PyUntypedArray {
        unsafe { &*(self as *const Self as *const PyUntypedArray) }
    }

    /// Turn `&PyArray<T,D>` into `Py<PyArray<T,D>>`,
    /// i.e. a pointer into Python's heap which is independent of the GIL lifetime.
    ///
    /// This method can be used to avoid lifetime annotations of function arguments
    /// or return values.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray1;
    /// use pyo3::{Py, Python};
    ///
    /// let array: Py<PyArray1<f64>> = Python::with_gil(|py| {
    ///     PyArray1::zeros(py, 5, false).to_owned()
    /// });
    ///
    /// Python::with_gil(|py| {
    ///     assert_eq!(array.as_ref(py).readonly().as_slice().unwrap(), [0.0; 5]);
    /// });
    /// ```
    pub fn to_owned(&self) -> Py<Self> {
        unsafe { Py::from_borrowed_ptr(self.py(), self.as_ptr()) }
    }

    /// Constructs a reference to a `PyArray` from a raw pointer to a Python object.
    ///
    /// # Safety
    ///
    /// This is a wrapper around [`pyo3::FromPyPointer::from_owned_ptr_or_opt`] and inherits its safety contract.
    pub unsafe fn from_owned_ptr<'py>(py: Python<'py>, ptr: *mut ffi::PyObject) -> &'py Self {
        py.from_owned_ptr(ptr)
    }

    /// Constructs a reference to a `PyArray` from a raw point to a Python object.
    ///
    /// # Safety
    ///
    /// This is a wrapper around [`pyo3::FromPyPointer::from_borrowed_ptr_or_opt`] and inherits its safety contract.
    pub unsafe fn from_borrowed_ptr<'py>(py: Python<'py>, ptr: *mut ffi::PyObject) -> &'py Self {
        py.from_borrowed_ptr(ptr)
    }

    /// Returns a pointer to the first element of the array.
    #[inline(always)]
    pub fn data(&self) -> *mut T {
        unsafe { (*self.as_array_ptr()).data as *mut _ }
    }
}

impl<T: Element, D: Dimension> PyArray<T, D> {
    fn extract<'py, E>(ob: &'py PyAny) -> Result<&'py Self, E>
    where
        E: From<PyDowncastError<'py>> + From<DimensionalityError> + From<TypeError<'py>>,
    {
        // Check if the object is an array.
        let array = unsafe {
            if npyffi::PyArray_Check(ob.py(), ob.as_ptr()) == 0 {
                return Err(PyDowncastError::new(ob, Self::NAME).into());
            }
            &*(ob as *const PyAny as *const Self)
        };

        // Check if the dimensionality matches `D`.
        let src_ndim = array.ndim();
        if let Some(dst_ndim) = D::NDIM {
            if src_ndim != dst_ndim {
                return Err(DimensionalityError::new(src_ndim, dst_ndim).into());
            }
        }

        // Check if the element type matches `T`.
        let src_dtype = array.dtype();
        let dst_dtype = T::get_dtype(ob.py());
        if !src_dtype.is_equiv_to(dst_dtype) {
            return Err(TypeError::new(src_dtype, dst_dtype).into());
        }

        Ok(array)
    }

    /// Same as [`shape`][PyUntypedArray::shape], but returns `D` instead of `&[usize]`.
    #[inline(always)]
    pub fn dims(&self) -> D {
        D::from_dimension(&Dim(self.shape())).expect(DIMENSIONALITY_MISMATCH_ERR)
    }

    /// Creates a new uninitialized NumPy array.
    ///
    /// If `is_fortran` is true, then it has Fortran/column-major order,
    /// otherwise it has C/row-major order.
    ///
    /// # Safety
    ///
    /// The returned array will always be safe to be dropped as the elements must either
    /// be trivially copyable (as indicated by `<T as Element>::IS_COPY`) or be pointers
    /// into Python's heap, which NumPy will automatically zero-initialize.
    ///
    /// However, the elements themselves will not be valid and should be initialized manually
    /// using raw pointers obtained via [`uget_raw`][Self::uget_raw]. Before that, all methods
    /// which produce references to the elements invoke undefined behaviour. In particular,
    /// zero-initialized pointers are _not_ valid instances of `PyObject`.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray3;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let arr = unsafe {
    ///         let arr = PyArray3::<i32>::new(py, [4, 5, 6], false);
    ///
    ///         for i in 0..4 {
    ///             for j in 0..5 {
    ///                 for k in 0..6 {
    ///                     arr.uget_raw([i, j, k]).write((i * j * k) as i32);
    ///                 }
    ///             }
    ///         }
    ///
    ///         arr
    ///     };
    ///
    ///     assert_eq!(arr.shape(), &[4, 5, 6]);
    /// });
    /// ```
    pub unsafe fn new<'py, ID>(py: Python<'py>, dims: ID, is_fortran: bool) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let flags = c_int::from(is_fortran);
        Self::new_uninit(py, dims, ptr::null_mut(), flags)
    }

    pub(crate) unsafe fn new_uninit<'py, ID>(
        py: Python<'py>,
        dims: ID,
        strides: *const npy_intp,
        flag: c_int,
    ) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let mut dims = dims.into_dimension();
        let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
            T::get_dtype(py).into_dtype_ptr(),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            strides as *mut npy_intp, // strides
            ptr::null_mut(),          // data
            flag,                     // flag
            ptr::null_mut(),          // obj
        );

        Self::from_owned_ptr(py, ptr)
    }

    unsafe fn new_with_data<'py, ID>(
        py: Python<'py>,
        dims: ID,
        strides: *const npy_intp,
        data_ptr: *const T,
        container: *mut PyAny,
    ) -> &'py Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let mut dims = dims.into_dimension();
        let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
            T::get_dtype(py).into_dtype_ptr(),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            strides as *mut npy_intp,    // strides
            data_ptr as *mut c_void,     // data
            npyffi::NPY_ARRAY_WRITEABLE, // flag
            ptr::null_mut(),             // obj
        );

        PY_ARRAY_API.PyArray_SetBaseObject(
            py,
            ptr as *mut npyffi::PyArrayObject,
            container as *mut ffi::PyObject,
        );

        Self::from_owned_ptr(py, ptr)
    }

    pub(crate) unsafe fn from_raw_parts<'py>(
        py: Python<'py>,
        dims: D,
        strides: *const npy_intp,
        data_ptr: *const T,
        container: PySliceContainer,
    ) -> &'py Self {
        let container = PyClassInitializer::from(container)
            .create_cell(py)
            .expect("Failed to create slice container");

        Self::new_with_data(py, dims, strides, data_ptr, container as *mut PyAny)
    }

    /// Creates a NumPy array backed by `array` and ties its ownership to the Python object `container`.
    ///
    /// # Safety
    ///
    /// `container` is set as a base object of the returned array which must not be dropped until `container` is dropped.
    /// Furthermore, `array` must not be reallocated from the time this method is called and until `container` is dropped.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use pyo3::prelude::*;
    /// # use numpy::{ndarray::Array1, PyArray1};
    /// #
    /// #[pyclass]
    /// struct Owner {
    ///     array: Array1<f64>,
    /// }
    ///
    /// #[pymethods]
    /// impl Owner {
    ///     #[getter]
    ///     fn array<'py>(this: &'py PyCell<Self>) -> &'py PyArray1<f64> {
    ///         let array = &this.borrow().array;
    ///
    ///         // SAFETY: The memory backing `array` will stay valid as long as this object is alive
    ///         // as we do not modify `array` in any way which would cause it to be reallocated.
    ///         unsafe { PyArray1::borrow_from_array(array, this) }
    ///     }
    /// }
    /// ```
    pub unsafe fn borrow_from_array<'py, S>(
        array: &ArrayBase<S, D>,
        container: &'py PyAny,
    ) -> &'py Self
    where
        S: Data<Elem = T>,
    {
        let (strides, dims) = (array.npy_strides(), array.raw_dim());
        let data_ptr = array.as_ptr();

        let py = container.py();

        mem::forget(container.to_object(py));

        Self::new_with_data(
            py,
            dims,
            strides.as_ptr(),
            data_ptr,
            container as *const PyAny as *mut PyAny,
        )
    }

    /// Construct a new NumPy array filled with zeros.
    ///
    /// If `is_fortran` is true, then it has Fortran/column-major order,
    /// otherwise it has C/row-major order.
    ///
    /// For arrays of Python objects, this will fill the array
    /// with valid pointers to zero-valued Python integer objects.
    ///
    /// See also [`numpy.zeros`][numpy-zeros] and [`PyArray_Zeros`][PyArray_Zeros].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray2;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray: &PyArray2<usize> = PyArray2::zeros(py, [2, 2], true);
    ///
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), [0; 4]);
    /// });
    /// ```
    ///
    /// [numpy-zeros]: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    /// [PyArray_Zeros]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Zeros
    pub fn zeros<'py, ID>(py: Python<'py>, dims: ID, is_fortran: bool) -> &Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let mut dims = dims.into_dimension();
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_Zeros(
                py,
                dims.ndim_cint(),
                dims.as_dims_ptr(),
                T::get_dtype(py).into_dtype_ptr(),
                if is_fortran { -1 } else { 0 },
            );
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// Returns an immutable view of the internal data as a slice.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased mutably by other instances of `PyArray`
    /// or concurrently modified by Python or other native code.
    ///
    /// Please consider the safe alternative [`PyReadonlyArray::as_slice`].
    pub unsafe fn as_slice(&self) -> Result<&[T], NotContiguousError> {
        if self.is_contiguous() {
            Ok(slice::from_raw_parts(self.data(), self.len()))
        } else {
            Err(NotContiguousError)
        }
    }

    /// Returns a mutable view of the internal data as a slice.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased immutably or mutably by other instances of [`PyArray`]
    /// or concurrently modified by Python or other native code.
    ///
    /// Please consider the safe alternative [`PyReadwriteArray::as_slice_mut`].
    pub unsafe fn as_slice_mut(&self) -> Result<&mut [T], NotContiguousError> {
        if self.is_contiguous() {
            Ok(slice::from_raw_parts_mut(self.data(), self.len()))
        } else {
            Err(NotContiguousError)
        }
    }

    /// Constructs a NumPy from an [`ndarray::Array`]
    ///
    /// This method uses the internal [`Vec`] of the [`ndarray::Array`] as the base object of the NumPy array.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use ndarray::array;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::from_owned_array(py, array![[1, 2], [3, 4]]);
    ///
    ///     assert_eq!(pyarray.readonly().as_array(), array![[1, 2], [3, 4]]);
    /// });
    /// ```
    pub fn from_owned_array<'py>(py: Python<'py>, mut arr: Array<T, D>) -> &'py Self {
        let (strides, dims) = (arr.npy_strides(), arr.raw_dim());
        let data_ptr = arr.as_mut_ptr();
        unsafe {
            Self::from_raw_parts(
                py,
                dims,
                strides.as_ptr(),
                data_ptr,
                PySliceContainer::from(arr),
            )
        }
    }

    /// Get a reference of the specified element if the given index is valid.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased mutably by other instances of `PyArray`
    /// or concurrently modified by Python or other native code.
    ///
    /// Consider using safe alternatives like [`PyReadonlyArray::get`].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap();
    ///
    ///     assert_eq!(unsafe { *pyarray.get([1, 0, 3]).unwrap() }, 11);
    /// });
    /// ```
    #[inline(always)]
    pub unsafe fn get(&self, index: impl NpyIndex<Dim = D>) -> Option<&T> {
        let ptr = self.get_raw(index)?;
        Some(&*ptr)
    }

    /// Same as [`get`][Self::get], but returns `Option<&mut T>`.
    ///
    /// # Safety
    ///
    /// Calling this method is undefined behaviour if the underlying array
    /// is aliased immutably or mutably by other instances of [`PyArray`]
    /// or concurrently modified by Python or other native code.
    ///
    /// Consider using safe alternatives like [`PyReadwriteArray::get_mut`].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap();
    ///
    ///     unsafe {
    ///         *pyarray.get_mut([1, 0, 3]).unwrap() = 42;
    ///     }
    ///
    ///     assert_eq!(unsafe { *pyarray.get([1, 0, 3]).unwrap() }, 42);
    /// });
    /// ```
    #[inline(always)]
    pub unsafe fn get_mut(&self, index: impl NpyIndex<Dim = D>) -> Option<&mut T> {
        let ptr = self.get_raw(index)?;
        Some(&mut *ptr)
    }

    #[inline(always)]
    fn get_raw<Idx>(&self, index: Idx) -> Option<*mut T>
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_checked::<T>(self.shape(), self.strides())?;
        Some(unsafe { self.data().offset(offset) })
    }

    /// Get an immutable reference of the specified element,
    /// without checking the given index.
    ///
    /// See [`NpyIndex`] for what types can be used as the index.
    ///
    /// # Safety
    ///
    /// Passing an invalid index is undefined behavior.
    /// The element must also have been initialized and
    /// all other references to it is must also be shared.
    ///
    /// See [`PyReadonlyArray::get`] for a safe alternative.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap();
    ///
    ///     assert_eq!(unsafe { *pyarray.uget([1, 0, 3]) }, 11);
    /// });
    /// ```
    #[inline(always)]
    pub unsafe fn uget<Idx>(&self, index: Idx) -> &T
    where
        Idx: NpyIndex<Dim = D>,
    {
        &*self.uget_raw(index)
    }

    /// Same as [`uget`](Self::uget), but returns `&mut T`.
    ///
    /// # Safety
    ///
    /// Passing an invalid index is undefined behavior.
    /// The element must also have been initialized and
    /// other references to it must not exist.
    ///
    /// See [`PyReadwriteArray::get_mut`] for a safe alternative.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn uget_mut<Idx>(&self, index: Idx) -> &mut T
    where
        Idx: NpyIndex<Dim = D>,
    {
        &mut *self.uget_raw(index)
    }

    /// Same as [`uget`][Self::uget], but returns `*mut T`.
    ///
    /// # Safety
    ///
    /// Passing an invalid index is undefined behavior.
    #[inline(always)]
    pub unsafe fn uget_raw<Idx>(&self, index: Idx) -> *mut T
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_unchecked::<T>(self.strides());
        self.data().offset(offset) as *mut _
    }

    /// Get a copy of the specified element in the array.
    ///
    /// See [`NpyIndex`] for what types can be used as the index.
    ///
    /// # Example
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 16, 1).reshape([2, 2, 4]).unwrap();
    ///
    ///     assert_eq!(pyarray.get_owned([1, 0, 3]), Some(11));
    /// });
    /// ```
    pub fn get_owned<Idx>(&self, index: Idx) -> Option<T>
    where
        Idx: NpyIndex<Dim = D>,
    {
        unsafe { self.get(index) }.cloned()
    }

    /// Turn an array with fixed dimensionality into one with dynamic dimensionality.
    pub fn to_dyn(&self) -> &PyArray<T, IxDyn> {
        unsafe { PyArray::from_borrowed_ptr(self.py(), self.as_ptr()) }
    }

    /// Returns a copy of the internal data of the array as a [`Vec`].
    ///
    /// Fails if the internal array is not contiguous. See also [`as_slice`][Self::as_slice].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray2;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray= py
    ///         .eval("__import__('numpy').array([[0, 1], [2, 3]], dtype='int64')", None, None)
    ///         .unwrap()
    ///         .downcast::<PyArray2<i64>>()
    ///         .unwrap();
    ///
    ///     assert_eq!(pyarray.to_vec().unwrap(), vec![0, 1, 2, 3]);
    /// });
    /// ```
    pub fn to_vec(&self) -> Result<Vec<T>, NotContiguousError> {
        unsafe { self.as_slice() }.map(ToOwned::to_owned)
    }

    /// Construct a NumPy array from a [`ndarray::ArrayBase`].
    ///
    /// This method allocates memory in Python's heap via the NumPy API,
    /// and then copies all elements of the array there.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use ndarray::array;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::from_array(py, &array![[1, 2], [3, 4]]);
    ///
    ///     assert_eq!(pyarray.readonly().as_array(), array![[1, 2], [3, 4]]);
    /// });
    /// ```
    pub fn from_array<'py, S>(py: Python<'py>, arr: &ArrayBase<S, D>) -> &'py Self
    where
        S: Data<Elem = T>,
    {
        ToPyArray::to_pyarray(arr, py)
    }

    /// Get an immutable borrow of the NumPy array
    pub fn try_readonly(&self) -> Result<PyReadonlyArray<'_, T, D>, BorrowError> {
        PyReadonlyArray::try_new(self)
    }

    /// Get an immutable borrow of the NumPy array
    ///
    /// # Panics
    ///
    /// Panics if the allocation backing the array is currently mutably borrowed.
    ///
    /// For a non-panicking variant, use [`try_readonly`][Self::try_readonly].
    pub fn readonly(&self) -> PyReadonlyArray<'_, T, D> {
        self.try_readonly().unwrap()
    }

    /// Get a mutable borrow of the NumPy array
    pub fn try_readwrite(&self) -> Result<PyReadwriteArray<'_, T, D>, BorrowError> {
        PyReadwriteArray::try_new(self)
    }

    /// Get a mutable borrow of the NumPy array
    ///
    /// # Panics
    ///
    /// Panics if the allocation backing the array is currently borrowed or
    /// if the array is [flagged as][flags] not writeable.
    ///
    /// For a non-panicking variant, use [`try_readwrite`][Self::try_readwrite].
    ///
    /// [flags]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
    pub fn readwrite(&self) -> PyReadwriteArray<'_, T, D> {
        self.try_readwrite().unwrap()
    }

    fn as_view<S: RawData, F>(&self, from_shape_ptr: F) -> ArrayBase<S, D>
    where
        F: FnOnce(StrideShape<D>, *mut T) -> ArrayBase<S, D>,
    {
        fn inner<D: Dimension>(
            shape: &[usize],
            strides: &[isize],
            itemsize: usize,
            mut data_ptr: *mut u8,
        ) -> (StrideShape<D>, u32, *mut u8) {
            let shape = D::from_dimension(&Dim(shape)).expect(DIMENSIONALITY_MISMATCH_ERR);

            assert!(strides.len() <= 32, "{}", MAX_DIMENSIONALITY_ERR);

            let mut new_strides = D::zeros(strides.len());
            let mut inverted_axes = 0_u32;

            for i in 0..strides.len() {
                // FIXME(kngwyu): Replace this hacky negative strides support with
                // a proper constructor, when it's implemented.
                // See https://github.com/rust-ndarray/ndarray/issues/842 for more.
                if strides[i] >= 0 {
                    new_strides[i] = strides[i] as usize / itemsize;
                } else {
                    // Move the pointer to the start position.
                    data_ptr = unsafe { data_ptr.offset(strides[i] * (shape[i] as isize - 1)) };

                    new_strides[i] = (-strides[i]) as usize / itemsize;
                    inverted_axes |= 1 << i;
                }
            }

            (shape.strides(new_strides), inverted_axes, data_ptr)
        }

        let (shape, mut inverted_axes, data_ptr) = inner(
            self.shape(),
            self.strides(),
            mem::size_of::<T>(),
            self.data() as _,
        );

        let mut array = from_shape_ptr(shape, data_ptr as _);

        while inverted_axes != 0 {
            let axis = inverted_axes.trailing_zeros() as usize;
            inverted_axes &= !(1 << axis);

            array.invert_axis(Axis(axis));
        }

        array
    }

    /// Returns an [`ArrayView`] of the internal array.
    ///
    /// See also [`PyReadonlyArray::as_array`].
    ///
    /// # Safety
    ///
    /// Calling this method invalidates all exclusive references to the internal data, e.g. `&mut [T]` or `ArrayViewMut`.
    pub unsafe fn as_array(&self) -> ArrayView<'_, T, D> {
        self.as_view(|shape, ptr| ArrayView::from_shape_ptr(shape, ptr))
    }

    /// Returns an [`ArrayViewMut`] of the internal array.
    ///
    /// See also [`PyReadwriteArray::as_array_mut`].
    ///
    /// # Safety
    ///
    /// Calling this method invalidates all other references to the internal data, e.g. `ArrayView` or `ArrayViewMut`.
    pub unsafe fn as_array_mut(&self) -> ArrayViewMut<'_, T, D> {
        self.as_view(|shape, ptr| ArrayViewMut::from_shape_ptr(shape, ptr))
    }

    /// Returns the internal array as [`RawArrayView`] enabling element access via raw pointers
    pub fn as_raw_array(&self) -> RawArrayView<T, D> {
        self.as_view(|shape, ptr| unsafe { RawArrayView::from_shape_ptr(shape, ptr) })
    }

    /// Returns the internal array as [`RawArrayViewMut`] enabling element access via raw pointers
    pub fn as_raw_array_mut(&self) -> RawArrayViewMut<T, D> {
        self.as_view(|shape, ptr| unsafe { RawArrayViewMut::from_shape_ptr(shape, ptr) })
    }

    /// Get a copy of the array as an [`ndarray::Array`].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use ndarray::array;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 0, 4, 1).reshape([2, 2]).unwrap();
    ///
    ///     assert_eq!(
    ///         pyarray.to_owned_array(),
    ///         array![[0, 1], [2, 3]]
    ///     )
    /// });
    /// ```
    pub fn to_owned_array(&self) -> Array<T, D> {
        unsafe { self.as_array() }.to_owned()
    }
}

#[cfg(feature = "nalgebra")]
impl<N, D> PyArray<N, D>
where
    N: nalgebra::Scalar + Element,
    D: Dimension,
{
    fn try_as_matrix_shape_strides<R, C, RStride, CStride>(
        &self,
    ) -> Option<((R, C), (RStride, CStride))>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        let ndim = self.ndim();
        let shape = self.shape();
        let strides = self.strides();

        if ndim != 1 && ndim != 2 {
            return None;
        }

        if strides.iter().any(|strides| *strides < 0) {
            return None;
        }

        let rows = shape[0];
        let cols = *shape.get(1).unwrap_or(&1);

        if R::try_to_usize().map(|expected| rows == expected) == Some(false) {
            return None;
        }

        if C::try_to_usize().map(|expected| cols == expected) == Some(false) {
            return None;
        }

        let row_stride = strides[0] as usize / mem::size_of::<N>();
        let col_stride = strides
            .get(1)
            .map_or(rows, |stride| *stride as usize / mem::size_of::<N>());

        if RStride::try_to_usize().map(|expected| row_stride == expected) == Some(false) {
            return None;
        }

        if CStride::try_to_usize().map(|expected| col_stride == expected) == Some(false) {
            return None;
        }

        let shape = (R::from_usize(rows), C::from_usize(cols));

        let strides = (
            RStride::from_usize(row_stride),
            CStride::from_usize(col_stride),
        );

        Some((shape, strides))
    }

    /// Try to convert this array into a [`nalgebra::MatrixView`] using the given shape and strides.
    ///
    /// # Safety
    ///
    /// Calling this method invalidates all exclusive references to the internal data, e.g. `ArrayViewMut` or `MatrixSliceMut`.
    #[doc(alias = "nalgebra")]
    pub unsafe fn try_as_matrix<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixView<'_, N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        let (shape, strides) = self.try_as_matrix_shape_strides()?;

        let storage = nalgebra::ViewStorage::from_raw_parts(self.data(), shape, strides);

        Some(nalgebra::Matrix::from_data(storage))
    }

    /// Try to convert this array into a [`nalgebra::MatrixViewMut`] using the given shape and strides.
    ///
    /// # Safety
    ///
    /// Calling this method invalidates all other references to the internal data, e.g. `ArrayView`, `MatrixSlice`, `ArrayViewMut` or `MatrixSliceMut`.
    #[doc(alias = "nalgebra")]
    pub unsafe fn try_as_matrix_mut<R, C, RStride, CStride>(
        &self,
    ) -> Option<nalgebra::MatrixViewMut<'_, N, R, C, RStride, CStride>>
    where
        R: nalgebra::Dim,
        C: nalgebra::Dim,
        RStride: nalgebra::Dim,
        CStride: nalgebra::Dim,
    {
        let (shape, strides) = self.try_as_matrix_shape_strides()?;

        let storage = nalgebra::ViewStorageMut::from_raw_parts(self.data(), shape, strides);

        Some(nalgebra::Matrix::from_data(storage))
    }
}

impl<D: Dimension> PyArray<PyObject, D> {
    /// Construct a NumPy array containing objects stored in a [`ndarray::Array`]
    ///
    /// This method uses the internal [`Vec`] of the [`ndarray::Array`] as the base object of the NumPy array.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use pyo3::{pyclass, Py, Python};
    /// use numpy::PyArray;
    ///
    /// #[pyclass]
    /// struct CustomElement {
    ///     foo: i32,
    ///     bar: f64,
    /// }
    ///
    /// Python::with_gil(|py| {
    ///     let array = array![
    ///         Py::new(py, CustomElement {
    ///             foo: 1,
    ///             bar: 2.0,
    ///         }).unwrap(),
    ///         Py::new(py, CustomElement {
    ///             foo: 3,
    ///             bar: 4.0,
    ///         }).unwrap(),
    ///     ];
    ///
    ///     let pyarray = PyArray::from_owned_object_array(py, array);
    ///
    ///     assert!(pyarray.readonly().as_array().get(0).unwrap().as_ref(py).is_instance_of::<CustomElement>());
    /// });
    /// ```
    pub fn from_owned_object_array<'py, T>(py: Python<'py>, mut arr: Array<Py<T>, D>) -> &'py Self {
        let (strides, dims) = (arr.npy_strides(), arr.raw_dim());
        let data_ptr = arr.as_mut_ptr() as *const PyObject;
        unsafe {
            Self::from_raw_parts(
                py,
                dims,
                strides.as_ptr(),
                data_ptr,
                PySliceContainer::from(arr),
            )
        }
    }
}

impl<T: Copy + Element> PyArray<T, Ix0> {
    /// Get the single element of a zero-dimensional array.
    ///
    /// See [`inner`][crate::inner] for an example.
    pub fn item(&self) -> T {
        unsafe { *self.data() }
    }
}

impl<T: Element> PyArray<T, Ix1> {
    /// Construct a one-dimensional array from a [mod@slice].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let slice = &[1, 2, 3, 4, 5];
    ///     let pyarray = PyArray::from_slice(py, slice);
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// });
    /// ```
    pub fn from_slice<'py>(py: Python<'py>, slice: &[T]) -> &'py Self {
        unsafe {
            let array = PyArray::new(py, [slice.len()], false);
            let mut data_ptr = array.data();
            clone_elements(slice, &mut data_ptr);
            array
        }
    }

    /// Construct a one-dimensional array from a [`Vec<T>`][Vec].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let vec = vec![1, 2, 3, 4, 5];
    ///     let pyarray = PyArray::from_vec(py, vec);
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// });
    /// ```
    #[inline(always)]
    pub fn from_vec<'py>(py: Python<'py>, vec: Vec<T>) -> &'py Self {
        vec.into_pyarray(py)
    }

    /// Construct a one-dimensional array from an [`Iterator`].
    ///
    /// If no reliable [`size_hint`][Iterator::size_hint] is available,
    /// this method can allocate memory multiple times, which can hurt performance.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::from_iter(py, "abcde".chars().map(u32::from));
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), &[97, 98, 99, 100, 101]);
    /// });
    /// ```
    pub fn from_iter<'py, I>(py: Python<'py>, iter: I) -> &'py Self
    where
        I: IntoIterator<Item = T>,
    {
        let data = iter.into_iter().collect::<Vec<_>>();
        data.into_pyarray(py)
    }
}

impl<T: Element> PyArray<T, Ix2> {
    /// Construct a two-dimension array from a [`Vec<Vec<T>>`][Vec].
    ///
    /// This function checks all dimensions of the inner vectors and returns
    /// an error if they are not all equal.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    /// use ndarray::array;
    ///
    /// Python::with_gil(|py| {
    ///     let vec2 = vec![vec![11, 12], vec![21, 22]];
    ///     let pyarray = PyArray::from_vec2(py, &vec2).unwrap();
    ///     assert_eq!(pyarray.readonly().as_array(), array![[11, 12], [21, 22]]);
    ///
    ///     let ragged_vec2 = vec![vec![11, 12], vec![21]];
    ///     assert!(PyArray::from_vec2(py, &ragged_vec2).is_err());
    /// });
    /// ```
    pub fn from_vec2<'py>(py: Python<'py>, v: &[Vec<T>]) -> Result<&'py Self, FromVecError> {
        let len2 = v.first().map_or(0, |v| v.len());
        let dims = [v.len(), len2];
        // SAFETY: The result of `Self::new` is always safe to drop.
        unsafe {
            let array = Self::new(py, dims, false);
            let mut data_ptr = array.data();
            for v in v {
                if v.len() != len2 {
                    cold();
                    return Err(FromVecError::new(v.len(), len2));
                }
                clone_elements(v, &mut data_ptr);
            }
            Ok(array)
        }
    }
}

impl<T: Element> PyArray<T, Ix3> {
    /// Construct a three-dimensional array from a [`Vec<Vec<Vec<T>>>`][Vec].
    ///
    /// This function checks all dimensions of the inner vectors and returns
    /// an error if they are not all equal.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    /// use ndarray::array;
    ///
    /// Python::with_gil(|py| {
    ///     let vec3 = vec![
    ///         vec![vec![111, 112], vec![121, 122]],
    ///         vec![vec![211, 212], vec![221, 222]],
    ///     ];
    ///     let pyarray = PyArray::from_vec3(py, &vec3).unwrap();
    ///     assert_eq!(
    ///         pyarray.readonly().as_array(),
    ///         array![[[111, 112], [121, 122]], [[211, 212], [221, 222]]]
    ///     );
    ///
    ///     let ragged_vec3 = vec![
    ///         vec![vec![111, 112], vec![121, 122]],
    ///         vec![vec![211], vec![221, 222]],
    ///     ];
    ///     assert!(PyArray::from_vec3(py, &ragged_vec3).is_err());
    /// });
    /// ```
    pub fn from_vec3<'py>(py: Python<'py>, v: &[Vec<Vec<T>>]) -> Result<&'py Self, FromVecError> {
        let len2 = v.first().map_or(0, |v| v.len());
        let len3 = v.first().map_or(0, |v| v.first().map_or(0, |v| v.len()));
        let dims = [v.len(), len2, len3];
        // SAFETY: The result of `Self::new` is always safe to drop.
        unsafe {
            let array = Self::new(py, dims, false);
            let mut data_ptr = array.data();
            for v in v {
                if v.len() != len2 {
                    cold();
                    return Err(FromVecError::new(v.len(), len2));
                }
                for v in v {
                    if v.len() != len3 {
                        cold();
                        return Err(FromVecError::new(v.len(), len3));
                    }
                    clone_elements(v, &mut data_ptr);
                }
            }
            Ok(array)
        }
    }
}

impl<T: Element, D> PyArray<T, D> {
    /// Copies `self` into `other`, performing a data type conversion if necessary.
    ///
    /// See also [`PyArray_CopyInto`][PyArray_CopyInto].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray_f = PyArray::arange(py, 2.0, 5.0, 1.0);
    ///     let pyarray_i = unsafe { PyArray::<i64, _>::new(py, [3], false) };
    ///
    ///     assert!(pyarray_f.copy_to(pyarray_i).is_ok());
    ///
    ///     assert_eq!(pyarray_i.readonly().as_slice().unwrap(), &[2, 3, 4]);
    /// });
    /// ```
    ///
    /// [PyArray_CopyInto]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_CopyInto
    pub fn copy_to<U: Element>(&self, other: &PyArray<U, D>) -> PyResult<()> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_CopyInto(self.py(), other_ptr, self_ptr) };
        if result != -1 {
            Ok(())
        } else {
            Err(PyErr::fetch(self.py()))
        }
    }

    /// Cast the `PyArray<T>` to `PyArray<U>`, by allocating a new array.
    ///
    /// See also [`PyArray_CastToType`][PyArray_CastToType].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray_f = PyArray::arange(py, 2.0, 5.0, 1.0);
    ///
    ///     let pyarray_i = pyarray_f.cast::<i32>(false).unwrap();
    ///
    ///     assert_eq!(pyarray_i.readonly().as_slice().unwrap(), &[2, 3, 4]);
    /// });
    /// ```
    ///
    /// [PyArray_CastToType]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_CastToType
    pub fn cast<'py, U: Element>(&'py self, is_fortran: bool) -> PyResult<&'py PyArray<U, D>> {
        let ptr = unsafe {
            PY_ARRAY_API.PyArray_CastToType(
                self.py(),
                self.as_array_ptr(),
                U::get_dtype(self.py()).into_dtype_ptr(),
                if is_fortran { -1 } else { 0 },
            )
        };
        if !ptr.is_null() {
            Ok(unsafe { PyArray::<U, D>::from_owned_ptr(self.py(), ptr) })
        } else {
            Err(PyErr::fetch(self.py()))
        }
    }

    /// Construct a new array which has same values as self,
    /// but has different dimensions specified by `dims`
    /// and a possibly different memory order specified by `order`.
    ///
    /// See also [`numpy.reshape`][numpy-reshape] and [`PyArray_Newshape`][PyArray_Newshape].
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::{npyffi::NPY_ORDER, PyArray};
    /// use pyo3::Python;
    /// use ndarray::array;
    ///
    /// Python::with_gil(|py| {
    ///     let array =
    ///         PyArray::from_iter(py, 0..9).reshape_with_order([3, 3], NPY_ORDER::NPY_FORTRANORDER).unwrap();
    ///
    ///     assert_eq!(array.readonly().as_array(), array![[0, 3, 6], [1, 4, 7], [2, 5, 8]]);
    ///     assert!(array.is_fortran_contiguous());
    ///
    ///     assert!(array.reshape([5]).is_err());
    /// });
    /// ```
    ///
    /// [numpy-reshape]: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    /// [PyArray_Newshape]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Newshape
    pub fn reshape_with_order<'py, ID: IntoDimension>(
        &'py self,
        dims: ID,
        order: NPY_ORDER,
    ) -> PyResult<&'py PyArray<T, ID::Dim>> {
        let mut dims = dims.into_dimension();
        let mut dims = dims.to_npy_dims();
        let ptr = unsafe {
            PY_ARRAY_API.PyArray_Newshape(
                self.py(),
                self.as_array_ptr(),
                &mut dims as *mut npyffi::PyArray_Dims,
                order,
            )
        };
        if !ptr.is_null() {
            Ok(unsafe { PyArray::<T, ID::Dim>::from_owned_ptr(self.py(), ptr) })
        } else {
            Err(PyErr::fetch(self.py()))
        }
    }

    /// Special case of [`reshape_with_order`][Self::reshape_with_order] which keeps the memory order the same.
    #[inline(always)]
    pub fn reshape<'py, ID: IntoDimension>(
        &'py self,
        dims: ID,
    ) -> PyResult<&'py PyArray<T, ID::Dim>> {
        self.reshape_with_order(dims, NPY_ORDER::NPY_ANYORDER)
    }

    /// Extends or truncates the dimensions of an array.
    ///
    /// This method works only on [contiguous][PyUntypedArray::is_contiguous] arrays.
    /// Missing elements will be initialized as if calling [`zeros`][Self::zeros].
    ///
    /// See also [`ndarray.resize`][ndarray-resize] and [`PyArray_Resize`][PyArray_Resize].
    ///
    /// # Safety
    ///
    /// There should be no outstanding references (shared or exclusive) into the array
    /// as this method might re-allocate it and thereby invalidate all pointers into it.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::<f64, _>::zeros(py, (10, 10), false);
    ///     assert_eq!(pyarray.shape(), [10, 10]);
    ///
    ///     unsafe {
    ///         pyarray.resize((100, 100)).unwrap();
    ///     }
    ///     assert_eq!(pyarray.shape(), [100, 100]);
    /// });
    /// ```
    ///
    /// [ndarray-resize]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html
    /// [PyArray_Resize]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Resize
    pub unsafe fn resize<ID: IntoDimension>(&self, dims: ID) -> PyResult<()> {
        let mut dims = dims.into_dimension();
        let mut dims = dims.to_npy_dims();
        let res = PY_ARRAY_API.PyArray_Resize(
            self.py(),
            self.as_array_ptr(),
            &mut dims as *mut npyffi::PyArray_Dims,
            1,
            NPY_ORDER::NPY_ANYORDER,
        );
        if !res.is_null() {
            Ok(())
        } else {
            Err(PyErr::fetch(self.py()))
        }
    }
}

impl<T: Element + AsPrimitive<f64>> PyArray<T, Ix1> {
    /// Return evenly spaced values within a given interval.
    ///
    /// See [numpy.arange][numpy.arange] for the Python API and [PyArray_Arange][PyArray_Arange] for the C API.
    ///
    /// # Example
    ///
    /// ```
    /// use numpy::PyArray;
    /// use pyo3::Python;
    ///
    /// Python::with_gil(|py| {
    ///     let pyarray = PyArray::arange(py, 2.0, 4.0, 0.5);
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), &[2.0, 2.5, 3.0, 3.5]);
    ///
    ///     let pyarray = PyArray::arange(py, -2, 4, 3);
    ///     assert_eq!(pyarray.readonly().as_slice().unwrap(), &[-2, 1]);
    /// });
    /// ```
    ///
    /// [numpy.arange]: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    /// [PyArray_Arange]: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Arange
    pub fn arange<'py>(py: Python<'py>, start: T, stop: T, step: T) -> &Self {
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_Arange(
                py,
                start.as_(),
                stop.as_(),
                step.as_(),
                T::get_dtype(py).num(),
            );
            Self::from_owned_ptr(py, ptr)
        }
    }
}

unsafe fn clone_elements<T: Element>(elems: &[T], data_ptr: &mut *mut T) {
    if T::IS_COPY {
        ptr::copy_nonoverlapping(elems.as_ptr(), *data_ptr, elems.len());
        *data_ptr = data_ptr.add(elems.len());
    } else {
        for elem in elems {
            data_ptr.write(elem.clone());
            *data_ptr = data_ptr.add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;
    use pyo3::{py_run, types::PyList};

    #[test]
    fn test_dyn_to_owned_array() {
        Python::with_gil(|py| {
            let array = PyArray::from_vec2(py, &[vec![1, 2], vec![3, 4]])
                .unwrap()
                .to_dyn()
                .to_owned_array();

            assert_eq!(array, array![[1, 2], [3, 4]].into_dyn());
        });
    }

    #[test]
    fn test_hasobject_flag() {
        Python::with_gil(|py| {
            let array: &PyArray<PyObject, _> =
                PyArray1::from_slice(py, &[PyList::empty(py).into()]);

            py_run!(py, array, "assert array.dtype.hasobject");
        });
    }
}

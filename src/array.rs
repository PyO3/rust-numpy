//! Safe interface for NumPy ndarray

use ndarray::*;
use npyffi;
use pyo3::*;
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr::null_mut;

use super::error::ArrayCastError;
use super::*;

/// Interface for [NumPy ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).
pub struct PyArray<T>(PyObject, PhantomData<T>);

pyobject_native_type_convert!(
    PyArray<T>,
    *npyffi::PyArray_Type_Ptr,
    npyffi::PyArray_Check,
    T
);

pyobject_native_type_named!(PyArray<T>, T);

impl<'a, T> ::std::convert::From<&'a PyArray<T>> for &'a PyObjectRef {
    fn from(ob: &'a PyArray<T>) -> Self {
        unsafe { &*(ob as *const PyArray<T> as *const PyObjectRef) }
    }
}

impl<'a, T: TypeNum> FromPyObject<'a> for &'a PyArray<T> {
    // here we do type-check twice
    // 1. Checks if the object is PyArray
    // 2. Checks if the data type of the array is T
    fn extract(ob: &'a PyObjectRef) -> PyResult<Self> {
        let array = unsafe {
            if npyffi::PyArray_Check(ob.as_ptr()) == 0 {
                return Err(PyDowncastError.into());
            }
            &*(ob as *const PyObjectRef as *const PyArray<T>)
        };
        array
            .type_check()
            .map(|_| array)
            .map_err(|err| err.into_pyerr("FromPyObject::extract typecheck failed"))
    }
}

impl<T> IntoPyObject for PyArray<T> {
    fn into_object(self, _py: Python) -> PyObject {
        self.0
    }
}

impl<T> PyArray<T> {
    /// Gets a raw `PyArrayObject` pointer.
    pub fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.as_ptr() as _
    }

    /// Constructs `PyArray` from raw python object without incrementing reference counts.
    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut pyo3::ffi::PyObject) -> Self {
        let obj = PyObject::from_owned_ptr(py, ptr);
        PyArray(obj, PhantomData)
    }

    /// Constructs PyArray from raw python object and increments reference counts.
    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut pyo3::ffi::PyObject) -> Self {
        let obj = PyObject::from_borrowed_ptr(py, ptr);
        PyArray(obj, PhantomData)
    }

    /// Returns the number of dimensions in the array.
    ///
    /// Same as [numpy.ndarray.ndim](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let arr = PyArray::<f64>::new(&np, &[4, 5, 6]);
    /// assert_eq!(arr.ndim(), 3);
    /// # }
    /// ```
    // C API: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_NDIM
    pub fn ndim(&self) -> usize {
        let ptr = self.as_array_ptr();
        unsafe { (*ptr).nd as usize }
    }

    /// Returns a slice which contains how many bytes you need to jump to the next row.
    ///
    /// Same as [numpy.ndarray.strides](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html)
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let arr = PyArray::<f64>::new(&np, &[4, 5, 6]);
    /// assert_eq!(arr.strides(), &[240, 48, 8]);
    /// # }
    /// ```
    // C API: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_STRIDES
    pub fn strides(&self) -> &[isize] {
        let n = self.ndim();
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).strides;
            ::std::slice::from_raw_parts(p, n)
        }
    }

    /// Returns a slice which contains dimmensions of the array.
    ///
    /// Same as [numpy.ndarray.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let arr = PyArray::<f64>::new(&np, &[4, 5, 6]);
    /// assert_eq!(arr.shape(), &[4, 5, 6]);
    /// # }
    /// ```
    // C API: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_DIMS
    pub fn shape(&self) -> &[usize] {
        let n = self.ndim();
        let ptr = self.as_array_ptr();
        unsafe {
            let p = (*ptr).dimensions as *mut usize;
            ::std::slice::from_raw_parts(p, n)
        }
    }

    /// Same as [shape](./struct.PyArray.html#method.shape)
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape()
    }

    /// Calcurates the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape().iter().fold(1, |a, b| a * b)
    }
}

impl<T: TypeNum> PyArray<T> {
    /// Construct one-dimension PyArray from boxed slice.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let slice = vec![1, 2, 3, 4, 5].into_boxed_slice();
    /// let pyarray = PyArray::from_boxed_slice(&np, slice);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_boxed_slice(np: &PyArrayModule, v: Box<[T]>) -> PyArray<T> {
        IntoPyArray::into_pyarray(v, np)
    }

    /// Construct one-dimension PyArray from `impl IntoIterator`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// use std::collections::BTreeSet;
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let set: BTreeSet<u32> = [4, 3, 2, 5, 1].into_iter().cloned().collect();
    /// let pyarray = PyArray::from_iter(&np, set);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_iter(np: &PyArrayModule, i: impl IntoIterator<Item = T>) -> PyArray<T> {
        i.into_iter().collect::<Vec<_>>().into_pyarray(np)
    }

    /// Construct one-dimension PyArray from Vec.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray = PyArray::from_vec(&np, vec![1, 2, 3, 4, 5]);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_vec(np: &PyArrayModule, v: Vec<T>) -> PyArray<T> {
        IntoPyArray::into_pyarray(v, np)
    }

    /// Construct a two-dimension PyArray from `Vec<Vec<T>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns `ArrayCastError`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let vec2 = vec![vec![1, 2, 3]; 2];
    /// let pyarray = PyArray::from_vec2(&np, &vec2).unwrap();
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2, 3], [1, 2, 3]].into_dyn());
    /// assert!(PyArray::from_vec2(&np, &vec![vec![1], vec![2, 3]]).is_err());
    /// # }
    /// ```
    pub fn from_vec2(np: &PyArrayModule, v: &Vec<Vec<T>>) -> Result<PyArray<T>, ArrayCastError> {
        let last_len = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != last_len) {
            return Err(ArrayCastError::FromVec);
        }
        let dims = [v.len(), last_len];
        let flattend: Vec<_> = v.iter().cloned().flatten().collect();
        unsafe {
            let data = convert::into_raw(flattend);
            Ok(PyArray::new_(np, &dims, null_mut(), data))
        }
    }

    /// Construct a three-dimension PyArray from `Vec<Vec<Vec<T>>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns `ArrayCastError`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let vec2 = vec![vec![vec![1, 2]; 2]; 2];
    /// let pyarray = PyArray::from_vec3(&np, &vec2).unwrap();
    /// assert_eq!(
    ///     pyarray.as_array().unwrap(),
    ///     array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]].into_dyn()
    /// );
    /// assert!(PyArray::from_vec3(&np, &vec![vec![vec![1], vec![]]]).is_err());
    /// # }
    /// ```
    pub fn from_vec3(
        np: &PyArrayModule,
        v: &Vec<Vec<Vec<T>>>,
    ) -> Result<PyArray<T>, ArrayCastError> {
        let dim2 = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != dim2) {
            return Err(ArrayCastError::FromVec);
        }
        let dim3 = v.last().map_or(0, |v| v.last().map_or(0, |v| v.len()));
        if v.iter().any(|v| v.iter().any(|v| v.len() != dim3)) {
            return Err(ArrayCastError::FromVec);
        }
        let dims = [v.len(), dim2, dim3];
        let flattend: Vec<_> = v.iter().flat_map(|v| v.iter().cloned().flatten()).collect();
        unsafe {
            let data = convert::into_raw(flattend);
            Ok(PyArray::new_(np, &dims, null_mut(), data))
        }
    }

    /// Construct PyArray from ndarray::Array.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray = PyArray::from_ndarray(&np, array![[1, 2], [3, 4]]);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2], [3, 4]].into_dyn());
    /// # }
    /// ```
    pub fn from_ndarray<D>(np: &PyArrayModule, arr: Array<T, D>) -> PyArray<T>
    where
        D: Dimension,
    {
        IntoPyArray::into_pyarray(arr, np)
    }

    /// Returns the pointer to the first element of the inner array.
    unsafe fn data(&self) -> *mut T {
        let ptr = self.as_array_ptr();
        (*ptr).data as *mut T
    }

    fn ndarray_shape(&self) -> StrideShape<IxDyn> {
        // FIXME may be done more simply
        let shape: Shape<_> = Dim(self.shape()).into();
        let st: Vec<usize> = self
            .strides()
            .iter()
            .map(|&x| x as usize / ::std::mem::size_of::<T>())
            .collect();
        shape.strides(Dim(st))
    }

    fn typenum(&self) -> i32 {
        unsafe {
            let descr = (*self.as_array_ptr()).descr;
            (*descr).type_num
        }
    }

    /// Returns the scalar type of the array.
    pub fn data_type(&self) -> NpyDataType {
        NpyDataType::from_i32(self.typenum())
    }

    fn type_check(&self) -> Result<(), ArrayCastError> {
        let truth = self.typenum();
        if T::is_same_type(truth) {
            Ok(())
        } else {
            Err(ArrayCastError::to_rust(truth, T::npy_data_type()))
        }
    }

    /// Get data as a ndarray::ArrayView
    pub fn as_array(&self) -> Result<ArrayViewD<T>, ArrayCastError> {
        self.type_check()?;
        unsafe { Ok(ArrayView::from_shape_ptr(self.ndarray_shape(), self.data())) }
    }

    /// Get data as a ndarray::ArrayViewMut
    pub fn as_array_mut(&self) -> Result<ArrayViewMutD<T>, ArrayCastError> {
        self.type_check()?;
        unsafe {
            Ok(ArrayViewMut::from_shape_ptr(
                self.ndarray_shape(),
                self.data(),
            ))
        }
    }

    /// Get data as a Rust immutable slice
    pub fn as_slice(&self) -> Result<&[T], ArrayCastError> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts(self.data(), self.len())) }
    }

    /// Get data as a Rust mutable slice
    pub fn as_slice_mut(&self) -> Result<&mut [T], ArrayCastError> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts_mut(self.data(), self.len())) }
    }

    /// Construct a new PyArray given a raw pointer and dimensions.
    ///
    /// Please use `new` or from methods instead.
    pub unsafe fn new_(
        np: &PyArrayModule,
        dims: &[usize],
        strides: *mut npy_intp,
        data: *mut c_void,
    ) -> Self {
        let dims: Vec<_> = dims.iter().map(|d| *d as npy_intp).collect();
        let ptr = np.PyArray_New(
            np.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.len() as i32,
            dims.as_ptr() as *mut npy_intp,
            T::typenum_default(),
            strides,
            data,
            0,                      // itemsize
            0,                      // flag
            ::std::ptr::null_mut(), //obj
        );
        Self::from_owned_ptr(np.py(), ptr)
    }

    /// Creates a new uninitialized array.
    ///
    /// See also [PyArray_SimpleNew](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNew).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray = PyArray::<i32>::new(&np, &[4, 5, 6]);
    /// assert_eq!(pyarray.shape(), &[4, 5, 6]);
    /// # }
    /// ```
    pub fn new(np: &PyArrayModule, dims: &[usize]) -> Self {
        unsafe { Self::new_(np, dims, null_mut(), null_mut()) }
    }

    /// Construct a new nd-dimensional array filled with 0. If `is_fortran` is true, then
    /// a fortran order array is created, otherwise a C-order array is created.
    ///
    /// See also [PyArray_Zeros](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Zeros)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::{PyArray, PyArrayModule};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray = PyArray::zeros(&np, &[2, 2], false);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[0, 0], [0, 0]].into_dyn());
    /// # }
    /// ```
    pub fn zeros(np: &PyArrayModule, dims: &[usize], is_fortran: bool) -> Self {
        let dims: Vec<npy_intp> = dims.iter().map(|d| *d as npy_intp).collect();
        unsafe {
            let descr = np.PyArray_DescrFromType(T::typenum_default());
            let ptr = np.PyArray_Zeros(
                dims.len() as i32,
                dims.as_ptr() as *mut npy_intp,
                descr,
                if is_fortran { -1 } else { 0 },
            );
            Self::from_owned_ptr(np.py(), ptr)
        }
    }

    /// Return evenly spaced values within a given interval.
    /// Same as [numpy.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    ///
    /// See also [PyArray_Arange](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Arange).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray = PyArray::<f64>::arange(&np, 2.0, 4.0, 0.5);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[2.0, 2.5, 3.0, 3.5]);
    /// let pyarray = PyArray::<i32>::arange(&np, -2.0, 4.0, 3.0);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[-2, 1]);
    /// # }
    pub fn arange(np: &PyArrayModule, start: f64, stop: f64, step: f64) -> Self {
        unsafe {
            let ptr = np.PyArray_Arange(start, stop, step, T::typenum_default());
            Self::from_owned_ptr(np.py(), ptr)
        }
    }

    /// Copies self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray_f = PyArray::<f64>::arange(&np, 2.0, 5.0, 1.0);
    /// let mut pyarray_i = PyArray::<i64>::new(&np, &[3]);
    /// assert!(pyarray_f.copy_to(&np, &mut pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn copy_to<U: TypeNum>(
        &self,
        np: &PyArrayModule,
        other: &mut PyArray<U>,
    ) -> Result<(), ArrayCastError> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { np.PyArray_CopyInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(ArrayCastError::Numpy {
                from: T::npy_data_type(),
                to: U::npy_data_type(),
            })
        } else {
            Ok(())
        }
    }

    /// Move the data of self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray_f = PyArray::<f64>::arange(&np, 2.0, 5.0, 1.0);
    /// let mut pyarray_i = PyArray::<i64>::new(&np, &[3]);
    /// assert!(pyarray_f.move_to(&np, &mut pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn move_to<U: TypeNum>(
        self,
        np: &PyArrayModule,
        other: &mut PyArray<U>,
    ) -> Result<(), ArrayCastError> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { np.PyArray_MoveInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(ArrayCastError::Numpy {
                from: T::npy_data_type(),
                to: U::npy_data_type(),
            })
        } else {
            Ok(())
        }
    }

    /// Cast the `PyArray<T>` to `PyArray<U>`, by allocating a new array.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, PyArrayModule, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let np = PyArrayModule::import(gil.python()).unwrap();
    /// let pyarray_f = PyArray::<f64>::arange(&np, 2.0, 5.0, 1.0);
    /// let pyarray_i = pyarray_f.cast::<i32>(&np, false).unwrap();
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn cast<U: TypeNum>(
        &self,
        np: &PyArrayModule,
        is_fortran: bool,
    ) -> Result<PyArray<U>, ArrayCastError> {
        let ptr = unsafe {
            let descr = np.PyArray_DescrFromType(U::typenum_default());
            np.PyArray_CastToType(self.as_array_ptr(), descr, if is_fortran { -1 } else { 0 })
        };
        if ptr.is_null() {
            Err(ArrayCastError::Numpy {
                from: T::npy_data_type(),
                to: U::npy_data_type(),
            })
        } else {
            unsafe { Ok(PyArray::<U>::from_owned_ptr(self.py(), ptr)) }
        }
    }
}

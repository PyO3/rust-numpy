//! Safe interface for NumPy ndarray
use ndarray::*;
use npyffi::{self, npy_intp, PY_ARRAY_API};
use pyo3::*;
use std::iter::ExactSizeIterator;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

use convert::ToNpyDims;
use error::{ErrorKind, IntoPyErr};
use types::{NpyDataType, TypeNum, NPY_ORDER};

/// Interface for [NumPy ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).
pub struct PyArray<T>(PyObject, PhantomData<T>);

pub fn get_array_module(py: Python) -> PyResult<&PyModule> {
    PyModule::import(py, npyffi::array::MOD_NAME)
}

pyobject_native_type_convert!(
    PyArray<T>,
    *npyffi::PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
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

    /// Get `PyArray` from `&PyArray`, by increasing ref counts.
    ///
    /// You can use this method when you have to avoid lifetime annotation to your function args
    /// or return types, like used with pyo3's `pymethod`.
    ///
    /// Since this method increases refcount, you can use `PyArray` even after `pyo3::GILGuard`
    /// dropped, in most cases.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use pyo3::{GILGuard, Python};
    /// use numpy::PyArray;
    /// fn return_py_array() -> PyArray<i32> {
    ///    let gil = Python::acquire_gil();
    ///    let array = PyArray::zeros(gil.python(), [5], false);
    ///    array.to_owned(gil.python())
    /// }
    /// let array = return_py_array();
    /// assert_eq!(array.as_slice().unwrap(), &[0, 0, 0, 0, 0]);
    /// # }
    /// ```
    pub fn to_owned(&self, py: Python) -> Self {
        let obj = unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) };
        PyArray(obj, PhantomData)
    }

    /// Constructs `PyArray` from raw python object without incrementing reference counts.
    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut pyo3::ffi::PyObject) -> &Self {
        py.from_owned_ptr(ptr)
    }

    /// Constructs PyArray from raw python object and increments reference counts.
    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut pyo3::ffi::PyObject) -> &Self {
        py.from_owned_ptr(ptr)
    }

    /// Returns the number of dimensions in the array.
    ///
    /// Same as [numpy.ndarray.ndim](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::<f64>::new(gil.python(), [4, 5, 6], false);
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
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::<f64>::new(gil.python(), [4, 5, 6], false);
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
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::<f64>::new(gil.python(), [4, 5, 6], false);
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

    /// Returns the pointer to the first element of the inner array.
    unsafe fn data(&self) -> *mut T {
        let ptr = self.as_array_ptr();
        (*ptr).data as *mut T
    }

    // TODO: we should provide safe access API
    unsafe fn get_unchecked(&self, index: &[isize]) -> *const T {
        let size = mem::size_of::<T>() as isize;
        index
            .iter()
            .zip(self.strides())
            .fold(self.data(), |pointer, (idx, stride)| {
                pointer.offset(stride * idx / size)
            })
    }

    // TODO: we should provide safe access API
    #[inline(always)]
    unsafe fn get_unchecked_mut(&self, index: &[isize]) -> *mut T {
        self.get_unchecked(index) as *mut T
    }
}

impl<T: TypeNum> PyArray<T> {
    /// Construct one-dimension PyArray from slice.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let array = [1, 2, 3, 4, 5];
    /// let pyarray = PyArray::from_slice(gil.python(), &array);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_slice<'py>(py: Python<'py>, slice: &[T]) -> &'py Self {
        let array = PyArray::new(py, [slice.len()], false);
        unsafe {
            let src = slice.as_ptr() as *mut T;
            ptr::copy_nonoverlapping(src, array.data(), slice.len());
        }
        array
    }

    /// Construct one-dimension PyArray from `impl ExactSizeIterator`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// use std::collections::BTreeSet;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let pyarray = PyArray::from_iter(gil.python(), vec.iter().map(|&x| x));
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_exact_iter(py: Python, iter: impl ExactSizeIterator<Item = T>) -> &Self {
        let array = Self::new(py, [iter.len()], false);
        unsafe {
            for (i, item) in iter.enumerate() {
                *array.get_unchecked_mut(&[i as isize]) = item;
            }
        }
        array
    }

    /// Construct one-dimension PyArray from `impl IntoIterator`.
    ///
    /// This method can allocate multiple times and not fast.
    /// When you can use [from_exact_iter](method.from_exact_iter.html), please use it.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// use std::collections::BTreeSet;
    /// let gil = pyo3::Python::acquire_gil();
    /// let set: BTreeSet<u32> = [4, 3, 2, 5, 1].into_iter().cloned().collect();
    /// let pyarray = PyArray::from_iter(gil.python(), set);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    pub fn from_iter(py: Python, iter: impl IntoIterator<Item = T>) -> &Self {
        // ↓ max cached size of ndarray
        let mut capacity = 1024 / mem::size_of::<T>();
        let array = Self::new(py, [capacity], false);
        let mut length = 0;
        unsafe {
            for (i, item) in iter.into_iter().enumerate() {
                length += 1;
                if length >= capacity {
                    capacity *= 2;
                    array
                        .resize_([capacity], 0, NPY_ORDER::NPY_ANYORDER)
                        .expect("PyArray::from_iter: Failed to allocate memory");
                }
                *array.get_unchecked_mut(&[i as isize]) = item;
            }
        }
        if capacity > length {
            array.resize_([length], 0, NPY_ORDER::NPY_ANYORDER).unwrap()
        }
        array
    }

    /// Construct a two-dimension PyArray from `Vec<Vec<T>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns `ArrayCastError`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec2 = vec![vec![1, 2, 3]; 2];
    /// let pyarray = PyArray::from_vec2(gil.python(), &vec2).unwrap();
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2, 3], [1, 2, 3]].into_dyn());
    /// assert!(PyArray::from_vec2(gil.python(), &vec![vec![1], vec![2, 3]]).is_err());
    /// # }
    /// ```
    pub fn from_vec2<'py>(py: Python<'py>, v: &Vec<Vec<T>>) -> Result<&'py Self, ErrorKind>
    where
        T: Clone,
    {
        let last_len = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != last_len) {
            return Err(ErrorKind::FromVec {
                dim1: v.len(),
                dim2: last_len,
            });
        }
        let dims = [v.len(), last_len];
        let array = Self::new(py, dims, false);
        unsafe {
            for y in 0..v.len() {
                for x in 0..last_len {
                    *array.get_unchecked_mut(&[y as isize, x as isize]) = v[y][x].clone();
                }
            }
        }
        Ok(array)
    }

    /// Construct a three-dimension PyArray from `Vec<Vec<Vec<T>>>`.
    ///
    /// This function checks all dimension of inner vec, and if there's any vec
    /// where its dimension differs from others, it returns `ArrayCastError`.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let vec2 = vec![vec![vec![1, 2]; 2]; 2];
    /// let pyarray = PyArray::from_vec3(gil.python(), &vec2).unwrap();
    /// assert_eq!(
    ///     pyarray.as_array().unwrap(),
    ///     array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]].into_dyn()
    /// );
    /// assert!(PyArray::from_vec3(gil.python(), &vec![vec![vec![1], vec![]]]).is_err());
    /// # }
    /// ```
    pub fn from_vec3<'py>(
        py: Python<'py>,
        v: &Vec<Vec<Vec<T>>>,
    ) -> Result<&'py PyArray<T>, ErrorKind>
    where
        T: Clone,
    {
        let dim2 = v.last().map_or(0, |v| v.len());
        if v.iter().any(|v| v.len() != dim2) {
            return Err(ErrorKind::FromVec {
                dim1: v.len(),
                dim2,
            });
        }
        let dim3 = v.last().map_or(0, |v| v.last().map_or(0, |v| v.len()));
        if v.iter().any(|v| v.iter().any(|v| v.len() != dim3)) {
            return Err(ErrorKind::FromVec {
                dim1: v.len(),
                dim2: dim3,
            });
        }
        let dims = [v.len(), dim2, dim3];
        let array = Self::new(py, dims, false);
        unsafe {
            for z in 0..v.len() {
                for y in 0..dim2 {
                    for x in 0..dim3 {
                        *array.get_unchecked_mut(&[z as isize, y as isize, x as isize]) =
                            v[z][y][x].clone();
                    }
                }
            }
        }
        Ok(array)
    }

    /// Construct PyArray from ndarray::Array.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::from_ndarray(gil.python(), array![[1, 2], [3, 4]]);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2], [3, 4]].into_dyn());
    /// # }
    /// ```
    pub fn from_ndarray<'py, S, D>(py: Python<'py>, arr: &ArrayBase<S, D>) -> &'py Self
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        let dims: Vec<_> = arr.shape().iter().cloned().collect();
        let len = dims.iter().fold(1, |prod, d| prod * d);
        let mut strides: Vec<_> = arr
            .strides()
            .into_iter()
            .map(|n| n * mem::size_of::<T>() as npy_intp)
            .collect();
        unsafe {
            let array = PyArray::new_(py, &*dims, strides.as_mut_ptr() as *mut npy_intp, 0);
            ptr::copy_nonoverlapping(arr.as_ptr(), array.data(), len);
            array
        }
    }

    /// Returns the scalar type of the array.
    pub fn data_type(&self) -> NpyDataType {
        NpyDataType::from_i32(self.typenum())
    }

    fn type_check(&self) -> Result<(), ErrorKind> {
        let truth = self.typenum();
        if T::is_same_type(truth) {
            Ok(())
        } else {
            Err(ErrorKind::to_rust(truth, T::npy_data_type()))
        }
    }

    /// Get data as a ndarray::ArrayView
    pub fn as_array(&self) -> Result<ArrayViewD<T>, ErrorKind> {
        self.type_check()?;
        unsafe { Ok(ArrayView::from_shape_ptr(self.ndarray_shape(), self.data())) }
    }

    /// Get data as a ndarray::ArrayViewMut
    pub fn as_array_mut(&self) -> Result<ArrayViewMutD<T>, ErrorKind> {
        self.type_check()?;
        unsafe {
            Ok(ArrayViewMut::from_shape_ptr(
                self.ndarray_shape(),
                self.data(),
            ))
        }
    }

    /// Get data as a Rust immutable slice
    pub fn as_slice(&self) -> Result<&[T], ErrorKind> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts(self.data(), self.len())) }
    }

    /// Get data as a Rust mutable slice
    pub fn as_slice_mut(&self) -> Result<&mut [T], ErrorKind> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts_mut(self.data(), self.len())) }
    }

    /// Creates a new uninitialized PyArray in python heap.
    ///
    /// See also [PyArray_SimpleNew](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNew).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::<i32>::new(gil.python(), [4, 5, 6]);
    /// assert_eq!(pyarray.shape(), &[4, 5, 6]);
    /// # }
    /// ```
    pub fn new<'py, D: ToNpyDims>(py: Python<'py>, dims: D, is_fortran: bool) -> &'py Self {
        let flags = if is_fortran { 1 } else { 0 };
        unsafe { PyArray::new_(py, dims, ptr::null_mut(), flags) }
    }

    unsafe fn new_<'py, D: ToNpyDims>(
        py: Python<'py>,
        dims: D,
        strides: *mut npy_intp,
        flag: c_int,
    ) -> &'py Self {
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.dims_len(),
            dims.dims_ptr(),
            T::typenum_default(),
            strides,                // strides
            ptr::null_mut(),        // data
            0,                      // itemsize
            flag,                   // flag
            ::std::ptr::null_mut(), //obj
        );
        Self::from_owned_ptr(py, ptr)
    }

    /// Construct a new nd-dimensional array filled with 0. If `is_fortran` is true, then
    /// a fortran order array is created, otherwise a C-order array is created.
    ///
    /// See also [PyArray_Zeros](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Zeros)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::zeros(gil.python(), [2, 2], false);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[0, 0], [0, 0]].into_dyn());
    /// # }
    /// ```
    pub fn zeros<'py, D: ToNpyDims>(py: Python<'py>, dims: D, is_fortran: bool) -> &'py Self {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(T::typenum_default());
            let ptr = PY_ARRAY_API.PyArray_Zeros(
                dims.dims_len(),
                dims.dims_ptr(),
                descr,
                if is_fortran { -1 } else { 0 },
            );
            Self::from_owned_ptr(py, ptr)
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
    /// use numpy::{PyArray, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::<f64>::arange(gil.python(), 2.0, 4.0, 0.5);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[2.0, 2.5, 3.0, 3.5]);
    /// let pyarray = PyArray::<i32>::arange(gil.python(), -2.0, 4.0, 3.0);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[-2, 1]);
    /// # }
    pub fn arange<'py>(py: Python<'py>, start: f64, stop: f64, step: f64) -> &'py Self {
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_Arange(start, stop, step, T::typenum_default());
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// Copies self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::<f64>::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = PyArray::<i64>::new(gil.python(), [3]);
    /// assert!(pyarray_f.copy_to(pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn copy_to<U: TypeNum>(&self, other: &PyArray<U>) -> Result<(), ErrorKind> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_CopyInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(ErrorKind::dtype_cast(self, U::npy_data_type()))
        } else {
            Ok(())
        }
    }

    /// Move the data of self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::<f64>::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = PyArray::<i64>::new(gil.python(), [3]);
    /// assert!(pyarray_f.move_to(pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn move_to<U: TypeNum>(&self, other: &PyArray<U>) -> Result<(), ErrorKind> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_MoveInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(ErrorKind::dtype_cast(self, U::npy_data_type()))
        } else {
            Ok(())
        }
    }

    /// Cast the `PyArray<T>` to `PyArray<U>`, by allocating a new array.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::{PyArray, IntoPyArray};
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::<f64>::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = pyarray_f.cast::<i32>(false).unwrap();
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn cast<'py, U: TypeNum>(
        &'py self,
        is_fortran: bool,
    ) -> Result<&'py PyArray<U>, ErrorKind> {
        let ptr = unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(U::typenum_default());
            PY_ARRAY_API.PyArray_CastToType(
                self.as_array_ptr(),
                descr,
                if is_fortran { -1 } else { 0 },
            )
        };
        if ptr.is_null() {
            Err(ErrorKind::dtype_cast(self, U::npy_data_type()))
        } else {
            Ok(unsafe { PyArray::<U>::from_owned_ptr(self.py(), ptr) })
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
    /// # #[macro_use] extern crate ndarray; extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let array = PyArray::from_vec(gil.python(), (0..9).collect());
    /// let array = array.reshape([3, 3]).unwrap();
    /// assert_eq!(array.as_array().unwrap(), array![[0, 1, 2], [3, 4, 5], [6, 7, 8]].into_dyn());
    /// assert!(array.reshape([5]).is_err());
    /// # }
    /// ```
    #[inline(always)]
    pub fn reshape<'py, D: ToNpyDims>(&'py self, dims: D) -> Result<&Self, ErrorKind> {
        self.reshape_with_order(dims, NPY_ORDER::NPY_ANYORDER)
    }

    /// Same as [reshape](method.reshape.html), but you can change the order of returned matrix.
    pub fn reshape_with_order<'py, D: ToNpyDims>(
        &'py self,
        dims: D,
        order: NPY_ORDER,
    ) -> Result<&Self, ErrorKind> {
        let mut np_dims = dims.to_npy_dims();
        let ptr = unsafe {
            PY_ARRAY_API.PyArray_Newshape(
                self.as_array_ptr(),
                &mut np_dims as *mut npyffi::PyArray_Dims,
                order,
            )
        };
        if ptr.is_null() {
            Err(ErrorKind::dims_cast(self, dims))
        } else {
            Ok(unsafe { PyArray::<T>::from_owned_ptr(self.py(), ptr) })
        }
    }

    // TODO: expose?
    fn resize_<'py, D: ToNpyDims>(
        &'py self,
        dims: D,
        check_ref: c_int,
        order: NPY_ORDER,
    ) -> Result<(), ErrorKind> {
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
            Err(ErrorKind::dims_cast(self, dims))
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_get_unchecked() {
    let gil = pyo3::Python::acquire_gil();
    let array = PyArray::from_slice(gil.python(), &[1i32, 2, 3]);
    unsafe {
        assert_eq!(*array.get_unchecked(&[1]), 2);
    }
}

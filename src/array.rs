//! Safe interface for NumPy ndarray
use ndarray::*;
use npyffi::{self, npy_intp, NPY_ORDER, PY_ARRAY_API};
use num_traits::AsPrimitive;
use pyo3::*;
use std::iter::ExactSizeIterator;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_int;
use std::ptr;

use convert::{NpyIndex, ToNpyDims};
use error::{ErrorKind, IntoPyResult};
use types::{NpyDataType, TypeNum};

/// A safe, static-typed interface for
/// [NumPy ndarray](https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html).
///
/// # Memory location
/// Numpy api allows to use a memory area allocated outside Pyhton.
///
/// However, we designed `PyArray` to always **owns a memory area allocated in Python's private
/// heap**, where all memories are managed by GC.
///
/// This means you always need to pay allocation cost when you create a `PyArray`, but don't need
/// to fear memory leak.
///
/// # Reference
///
/// Like [`new`](#method.new), most constractor methods of this type returns `&PyArray`.
///
/// See [pyo3's document](https://pyo3.rs/master/doc/pyo3/index.html#ownership-and-lifetimes)
/// for the reason.
///
/// # Mutation
/// You can do destructive changes to `PyArray` via &self methods like [`move_to`](#method.move_to).
///
/// About this design, see
/// [pyo3's document](https://pyo3.rs/master/doc/pyo3/index.html#ownership-and-lifetimes), too.
///
/// # Dimension
/// `PyArray` has 2 type parametes `T` and `D`. `T` represents its data type like `f32`, and `D`
/// represents its dimension.
///
/// To specify the dimension, you can use types which implements
/// [Dimension](https://docs.rs/ndarray/0.12/ndarray/trait.Dimension.html).
///
/// Typically, you can use `Ix1, Ix2, ..` for fixed size arrays, and use `IxDyn` for dynamic
/// dimensioned arrays. They're re-exported from `ndarray` crate.
///
/// You can also use various type aliases we provide, like [`PyArray1`](./type.PyArray1.html)
/// or [`PyArrayDyn`](./type.PyArrayDyn.html).
///
/// Many constructor methods takes a type which implements
/// [`IntoDimension`](https://docs.rs/ndarray/0.12/ndarray/dimension/conversion/trait.IntoDimension.html)
/// trait. Typically, you can use array(e.g. `[3, 4, 5]`) or tuple(e.g. `(3, 4, 5)`) as a dimension.
/// # Example
/// ```
/// # #[macro_use] extern crate ndarray; extern crate pyo3; extern crate numpy; fn main() {
/// use pyo3::{GILGuard, Python};
/// use numpy::PyArray;
/// use ndarray::Array;
/// let gil = Python::acquire_gil();
/// let pyarray = PyArray::arange(gil.python(), 0., 4., 1.).reshape([2, 2]).unwrap();
/// let array = array![[3., 4.], [5., 6.]];
/// assert_eq!(
///     array.dot(&pyarray.as_array().unwrap()),
///     array![[8., 15.], [12., 23.]]
/// );
/// # }
/// ```
pub struct PyArray<T, D>(PyObject, PhantomData<T>, PhantomData<D>);

/// one-dimensional array
pub type PyArray1<T> = PyArray<T, Ix1>;
/// two-dimensional array
pub type PyArray2<T> = PyArray<T, Ix2>;
/// three-dimensional array
pub type PyArray3<T> = PyArray<T, Ix3>;
/// four-dimensional array
pub type PyArray4<T> = PyArray<T, Ix4>;
/// five-dimensional array
pub type PyArray5<T> = PyArray<T, Ix5>;
/// six-dimensional array
pub type PyArray6<T> = PyArray<T, Ix6>;
/// dynamic-dimensional array
pub type PyArrayDyn<T> = PyArray<T, IxDyn>;

/// Returns a array module.
pub fn get_array_module(py: Python) -> PyResult<&PyModule> {
    PyModule::import(py, npyffi::array::MOD_NAME)
}

pyobject_native_type_convert!(
    PyArray<T, D>,
    *npyffi::PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
    npyffi::PyArray_Check,
    T, D
);

pyobject_native_type_named!(PyArray<T, D>, T, D);

impl<'a, T, D> ::std::convert::From<&'a PyArray<T, D>> for &'a PyObjectRef {
    fn from(ob: &'a PyArray<T, D>) -> Self {
        unsafe { &*(ob as *const PyArray<T, D> as *const PyObjectRef) }
    }
}

impl<'a, T: TypeNum, D: Dimension> FromPyObject<'a> for &'a PyArray<T, D> {
    // here we do type-check twice
    // 1. Checks if the object is PyArray
    // 2. Checks if the data type of the array is T
    fn extract(ob: &'a PyObjectRef) -> PyResult<Self> {
        let array = unsafe {
            if npyffi::PyArray_Check(ob.as_ptr()) == 0 {
                return Err(PyDowncastError.into());
            }
            if let Some(ndim) = D::NDIM {
                let ptr = ob.as_ptr() as *mut npyffi::PyArrayObject;
                if (*ptr).nd as usize != ndim {
                    return Err(PyErr::new::<exc::TypeError, _>(format!(
                        "specified dim was {}, but actual dim was {}",
                        ndim,
                        (*ptr).nd
                    )));
                }
            }
            &*(ob as *const PyObjectRef as *const PyArray<T, D>)
        };
        array
            .type_check()
            .map(|_| array)
            .into_pyresult_with(|| "FromPyObject::extract typecheck failed")
    }
}

impl<T, D> IntoPyObject for PyArray<T, D> {
    fn into_object(self, _py: Python) -> PyObject {
        self.0
    }
}

impl<T, D> PyArray<T, D> {
    /// Gets a raw `PyArrayObject` pointer.
    pub fn as_array_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.as_ptr() as _
    }

    // TODO: 'increasing ref counts' is really collect approach for extension?
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
    /// use numpy::PyArray1;
    /// fn return_py_array() -> PyArray1<i32> {
    ///    let gil = Python::acquire_gil();
    ///    let array = PyArray1::zeros(gil.python(), [5], false);
    ///    array.to_owned(gil.python())
    /// }
    /// let array = return_py_array();
    /// assert_eq!(array.as_slice().unwrap(), &[0, 0, 0, 0, 0]);
    /// # }
    /// ```
    pub fn to_owned(&self, py: Python) -> Self {
        let obj = unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) };
        PyArray(obj, PhantomData, PhantomData)
    }

    /// Constructs `PyArray` from raw python object without incrementing reference counts.
    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut ffi::PyObject) -> &Self {
        py.from_owned_ptr(ptr)
    }

    /// Constructs PyArray from raw python object and increments reference counts.
    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut ffi::PyObject) -> &Self {
        py.from_borrowed_ptr(ptr)
    }

    /// Returns the number of dimensions in the array.
    ///
    /// Same as [numpy.ndarray.ndim](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
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
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
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
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray3::<f64>::new(gil.python(), [4, 5, 6], false);
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

    /// Calcurates the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape().iter().fold(1, |a, b| a * b)
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
}

impl<T: TypeNum, D: Dimension> PyArray<T, D> {
    /// Same as [shape](#method.shape), but returns `D`
    #[inline(always)]
    pub fn dims(&self) -> D {
        D::from_dimension(&Dim(self.shape())).expect("PyArray::dims different dimension")
    }

    fn ndarray_shape(&self) -> StrideShape<D> {
        // FIXME may be done more simply
        let shape: Shape<_> = Dim(self.dims()).into();
        let mut st = D::default();
        let size = mem::size_of::<T>();
        for (i, &s) in self.strides().iter().enumerate() {
            st[i] = s as usize / size;
        }
        shape.strides(st)
    }

    /// Creates a new uninitialized PyArray in python heap.
    ///
    /// If `is_fortran == true`, returns Fortran-order array. Else, returns C-order array.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray3;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray3::<i32>::new(gil.python(), [4, 5, 6], false);
    /// assert_eq!(pyarray.shape(), &[4, 5, 6]);
    /// # }
    /// ```
    pub fn new<'py, ID>(py: Python<'py>, dims: ID, is_fortran: bool) -> &'py Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let flags = if is_fortran { 1 } else { 0 };
        unsafe { PyArray::new_(py, dims, ptr::null_mut(), flags) }
    }

    unsafe fn new_<'py, ID>(
        py: Python<'py>,
        dims: ID,
        strides: *mut npy_intp,
        flag: c_int,
    ) -> &'py Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let dims = dims.into_dimension();
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            T::typenum_default(),
            strides,                // strides
            ptr::null_mut(),        // data
            0,                      // itemsize
            flag,                   // flag
            ::std::ptr::null_mut(), //obj
        );
        Self::from_owned_ptr(py, ptr)
    }

    /// Construct a new nd-dimensional array filled with 0.
    ///
    /// If `is_fortran` is true, then
    /// a fortran order array is created, otherwise a C-order array is created.
    ///
    /// See also [PyArray_Zeros](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Zeros)
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray2;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray2::zeros(gil.python(), [2, 2], false);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[0, 0], [0, 0]]);
    /// # }
    /// ```
    pub fn zeros<'py, ID>(py: Python<'py>, dims: ID, is_fortran: bool) -> &'py Self
    where
        ID: IntoDimension<Dim = D>,
    {
        let dims = dims.into_dimension();
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(T::typenum_default());
            let ptr = PY_ARRAY_API.PyArray_Zeros(
                dims.ndim_cint(),
                dims.as_dims_ptr(),
                descr,
                if is_fortran { -1 } else { 0 },
            );
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// Construct PyArray from ndarray::Array.
    ///
    /// This method allocates memory in Python's heap via numpy api, and then copies all elements
    /// of the array there.
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; #[macro_use] extern crate ndarray; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::from_ndarray(gil.python(), &array![[1, 2], [3, 4]]);
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2], [3, 4]]);
    /// # }
    /// ```
    pub fn from_ndarray<'py, S>(py: Python<'py>, arr: &ArrayBase<S, D>) -> &'py Self
    where
        S: Data<Elem = T>,
    {
        let len = arr.len();
        let mut strides: Vec<_> = arr
            .strides()
            .into_iter()
            .map(|n| n * mem::size_of::<T>() as npy_intp)
            .collect();
        unsafe {
            let array = PyArray::new_(py, arr.raw_dim(), strides.as_mut_ptr() as *mut npy_intp, 0);
            ptr::copy_nonoverlapping(arr.as_ptr(), array.data(), len);
            array
        }
    }

    /// Get the immutable view of the internal data of `PyArray`, as `ndarray::ArrayView`.
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate ndarray; extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py_array = PyArray::arange(gil.python(), 0, 4, 1).reshape([2, 2]).unwrap();
    /// assert_eq!(
    ///     py_array.as_array().unwrap(),
    ///     array![[0, 1], [2, 3]]
    /// )
    /// # }
    /// ```
    pub fn as_array(&self) -> Result<ArrayView<T, D>, ErrorKind> {
        self.type_check()?;
        unsafe { Ok(ArrayView::from_shape_ptr(self.ndarray_shape(), self.data())) }
    }

    /// Almost same as [`as_array`](#method.as_array), but returns `ArrayViewMut`.
    pub fn as_array_mut(&self) -> Result<ArrayViewMut<T, D>, ErrorKind> {
        self.type_check()?;
        unsafe {
            Ok(ArrayViewMut::from_shape_ptr(
                self.ndarray_shape(),
                self.data(),
            ))
        }
    }

    /// Get an immutable reference of a specified element, without checking the
    /// passed index is valid.
    ///
    /// See [NpyIndex](../convert/trait.NpyIndex.html) for what types you can use as index.
    ///
    /// Passing an invalid index can cause undefined behavior(mostly SIGSEGV).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::arange(gil.python(), 0, 16, 1).reshape([2, 2, 4]).unwrap();
    /// assert_eq!(*arr.get([1, 0, 3]).unwrap(), 11);
    /// assert!(arr.get([2, 0, 3]).is_none());
    /// # }
    /// ```
    ///
    /// For fixed dimension arrays, too long/short index causes compile error.
    /// ```compile_fail
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::arange(gil.python(), 0, 16, 1).reshape([2, 2, 4]).unwrap();
    /// let a = arr.get([1, 2]); // Compile Error!
    /// # }
    /// ```
    /// But, for dinamic errors too long/short index returns `None`.
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::arange(gil.python(), 0, 16, 1).reshape([2, 2, 4]).unwrap();
    /// let arr = arr.into_dyn();
    /// assert!(arr.get([1, 2].as_ref()).is_none());
    /// # }
    /// ```
    #[inline(always)]
    pub fn get<Idx>(&self, index: Idx) -> Option<&T>
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_checked::<T>(self.shape(), self.strides())?;
        unsafe { Some(&*self.data().offset(offset)) }
    }

    /// Same as [get](#method.get), but returns `&mut T`.
    #[inline(always)]
    pub fn get_mut<Idx>(&self, index: Idx) -> Option<&mut T>
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_checked::<T>(self.shape(), self.strides())?;
        unsafe { Some(&mut *(self.data().offset(offset) as *mut T)) }
    }

    /// Get an immutable reference of a specified element, without checking the
    /// passed index is valid.
    ///
    /// See [NpyIndex](../convert/trait.NpyIndex.html) for what types you can use as index.
    ///
    /// Passing an invalid index can cause undefined behavior(mostly SIGSEGV).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let arr = PyArray::arange(gil.python(), 0, 16, 1).reshape([2, 2, 4]).unwrap();
    /// assert_eq!(unsafe { *arr.uget([1, 0, 3]) }, 11);
    /// # }
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
    pub unsafe fn uget_mut<Idx>(&self, index: Idx) -> &mut T
    where
        Idx: NpyIndex<Dim = D>,
    {
        let offset = index.get_unchecked::<T>(self.strides());
        &mut *(self.data().offset(offset) as *mut T)
    }

    /// Get dynamic dimensioned array from fixed dimension array.
    ///
    /// See [get](#method.get) for usage.
    pub fn into_dyn(&self) -> &PyArray<T, IxDyn> {
        let python = self.py();
        unsafe { PyArray::from_borrowed_ptr(python, self.as_ptr()) }
    }
}

impl<T: TypeNum> PyArray<T, Ix1> {
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
                *array.uget_mut([i]) = item;
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
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::arange(gil.python(), 0, 10, 1);
    /// assert_eq!(pyarray.len(), 10);
    /// pyarray.resize(100).unwrap();
    /// assert_eq!(pyarray.len(), 100);
    /// # }
    /// ```
    pub fn resize(&self, new_elems: usize) -> Result<(), ErrorKind> {
        self.resize_([new_elems], 1, NPY_ORDER::NPY_ANYORDER)
    }

    fn resize_<'py, D: IntoDimension>(
        &'py self,
        dims: D,
        check_ref: c_int,
        order: NPY_ORDER,
    ) -> Result<(), ErrorKind> {
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
            Err(ErrorKind::dims_cast(self, dims))
        } else {
            Ok(())
        }
    }
}

impl<T: TypeNum> PyArray<T, Ix2> {
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
    /// assert_eq!(pyarray.as_array().unwrap(), array![[1, 2, 3], [1, 2, 3]]);
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
                    *array.uget_mut([y, x]) = v[y][x].clone();
                }
            }
        }
        Ok(array)
    }
}

impl<T: TypeNum> PyArray<T, Ix3> {
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
    ///     array![[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
    /// );
    /// assert!(PyArray::from_vec3(gil.python(), &vec![vec![vec![1], vec![]]]).is_err());
    /// # }
    /// ```
    pub fn from_vec3<'py>(py: Python<'py>, v: &Vec<Vec<Vec<T>>>) -> Result<&'py Self, ErrorKind>
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
                        *array.uget_mut([z, y, x]) = v[z][y][x].clone();
                    }
                }
            }
        }
        Ok(array)
    }
}

impl<T: TypeNum, D> PyArray<T, D> {
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

    /// Get the immutable view of the internal data of `PyArray`, as slice.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let py_array = PyArray::arange(gil.python(), 0, 4, 1).reshape([2, 2]).unwrap();
    /// assert_eq!(py_array.as_slice().unwrap(), &[0, 1, 2, 3]);
    /// # }
    /// ```
    pub fn as_slice(&self) -> Result<&[T], ErrorKind> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts(self.data(), self.len())) }
    }

    /// Get the mmutable view of the internal data of `PyArray`, as slice.
    pub fn as_slice_mut(&self) -> Result<&mut [T], ErrorKind> {
        self.type_check()?;
        unsafe { Ok(::std::slice::from_raw_parts_mut(self.data(), self.len())) }
    }

    /// Copies self into `other`, performing a data-type conversion if necessary.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = PyArray::<i64, _>::new(gil.python(), [3], false);
    /// assert!(pyarray_f.copy_to(pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn copy_to<U: TypeNum>(&self, other: &PyArray<U, D>) -> Result<(), ErrorKind> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_CopyInto(other_ptr, self_ptr) };
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
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::arange(gil.python(), 2.0, 5.0, 1.0);
    /// let pyarray_i = pyarray_f.cast::<i32>(false).unwrap();
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn cast<'py, U: TypeNum>(
        &'py self,
        is_fortran: bool,
    ) -> Result<&'py PyArray<U, D>, ErrorKind> {
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
    /// # #[macro_use] extern crate ndarray; extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let array = PyArray::from_exact_iter(gil.python(), 0..9);
    /// let array = array.reshape([3, 3]).unwrap();
    /// assert_eq!(array.as_array().unwrap(), array![[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
    /// assert!(array.reshape([5]).is_err());
    /// # }
    /// ```
    #[inline(always)]
    pub fn reshape<'py, ID, D2>(&'py self, dims: ID) -> Result<&'py PyArray<T, D2>, ErrorKind>
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
    ) -> Result<&'py PyArray<T, D2>, ErrorKind>
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
            Err(ErrorKind::dims_cast(self, dims))
        } else {
            Ok(unsafe { PyArray::<T, D2>::from_owned_ptr(self.py(), ptr) })
        }
    }
}

impl<T: TypeNum> PyArray<T, IxDyn> {
    /// Move the data of self into `other`, performing a data-type conversion if necessary.
    ///
    /// For type safety, you have to convert `PyArray` to `PyArrayDyn` before using this method.
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray_f = PyArray::arange(gil.python(), 2.0, 5.0, 1.0).into_dyn();
    /// let pyarray_i = PyArray::<i64, _>::new(gil.python(), [3], false);
    /// assert!(pyarray_f.move_to(pyarray_i).is_ok());
    /// assert_eq!(pyarray_i.as_slice().unwrap(), &[2, 3, 4]);
    /// # }
    pub fn move_to<U: TypeNum, D2: Dimension>(
        &self,
        other: &PyArray<U, D2>,
    ) -> Result<(), ErrorKind> {
        let self_ptr = self.as_array_ptr();
        let other_ptr = other.as_array_ptr();
        let result = unsafe { PY_ARRAY_API.PyArray_MoveInto(other_ptr, self_ptr) };
        if result == -1 {
            Err(ErrorKind::dtype_cast(self, U::npy_data_type()))
        } else {
            Ok(())
        }
    }
}

impl<T: TypeNum + AsPrimitive<f64>> PyArray<T, Ix1> {
    /// Return evenly spaced values within a given interval.
    /// Same as [numpy.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).
    ///
    /// See also [PyArray_Arange](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Arange).
    ///
    /// # Example
    /// ```
    /// # extern crate pyo3; extern crate numpy; fn main() {
    /// use numpy::PyArray;
    /// let gil = pyo3::Python::acquire_gil();
    /// let pyarray = PyArray::arange(gil.python(), 2.0, 4.0, 0.5);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[2.0, 2.5, 3.0, 3.5]);
    /// let pyarray = PyArray::arange(gil.python(), -2, 4, 3);
    /// assert_eq!(pyarray.as_slice().unwrap(), &[-2, 1]);
    /// # }
    pub fn arange<'py>(py: Python<'py>, start: T, stop: T, step: T) -> &'py Self {
        unsafe {
            let ptr = PY_ARRAY_API.PyArray_Arange(
                start.as_(),
                stop.as_(),
                step.as_(),
                T::typenum_default(),
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

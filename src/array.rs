
use npyffi;
use pyffi;
use cpython::*;
use npyffi::PyArrayModule;
use super::*;

pub struct PyArray(PyObject);

impl PyArray {
    pub fn as_ptr(&self) -> *mut npyffi::PyArrayObject {
        self.0.as_ptr() as *mut npyffi::PyArrayObject
    }

    pub fn steal_ptr(self) -> *mut npyffi::PyArrayObject {
        self.0.steal_ptr() as *mut npyffi::PyArrayObject
    }

    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut pyffi::PyObject) -> Self {
        let obj = PyObject::from_owned_ptr(py, ptr);
        PyArray(obj)
    }

    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut pyffi::PyObject) -> Self {
        let obj = PyObject::from_borrowed_ptr(py, ptr);
        PyArray(obj)
    }

    /// The number of dimensions in the array
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_NDIM
    pub fn ndim(&self) -> usize {
        let ptr = self.as_ptr();
        unsafe { (*ptr).nd as usize }
    }

    /// dimensions of the array
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_DIMS
    pub fn dims(&self) -> Vec<usize> {
        let n = self.ndim();
        let ptr = self.as_ptr();
        let dims = unsafe {
            let p = (*ptr).dimensions;
            ::std::slice::from_raw_parts(p, n)
        };
        dims.into_iter().map(|d| *d as usize).collect()
    }

    pub fn len(&self) -> usize {
        self.dims().iter().fold(1, |a, b| a * b)
    }

    /// A synonym for PyArray_DIMS, named to be consistent with the ‘shape’ usage within Python.
    pub fn shape(&self) -> Vec<usize> {
        self.dims()
    }

    /// The number of elements matches the number of dimensions of the array
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_STRIDES
    /// For the explaination of stride, see also https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html#numpy.ndarray.strides
    pub fn strides(&self) -> Vec<isize> {
        let n = self.ndim();
        let ptr = self.as_ptr();
        let dims = unsafe {
            let p = (*ptr).strides;
            ::std::slice::from_raw_parts(p, n)
        };
        dims.into_iter().map(|d| *d as isize).collect()
    }

    /// Get data as a Rust immutable slice
    pub fn as_slice<T>(&self) -> &[T] {
        let n = self.len();
        let ptr = self.as_ptr();
        unsafe {
            let p = (*ptr).data as *mut T;
            ::std::slice::from_raw_parts(p, n)
        }
    }

    /// Get data as a Rust mutable slice
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        let n = self.len();
        let ptr = self.as_ptr();
        unsafe {
            let p = (*ptr).data as *mut T;
            ::std::slice::from_raw_parts_mut(p, n)
        }
    }

    /// a wrapper of PyArray_SimpleNew
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNew
    pub fn new(py: Python, np: &PyArrayModule, dims: &[usize], typenum: NPY_TYPES) -> Self {
        let dims: Vec<npy_intp> = dims.iter().map(|d| *d as npy_intp).collect();
        unsafe {
            let ptr = np.PyArray_New(np.get_type_object(npyffi::ARRAY_TYPE::PyArray_Type),
                                     dims.len() as i32,
                                     dims.as_ptr() as *mut npy_intp,
                                     typenum as i32,
                                     ::std::ptr::null_mut(),
                                     ::std::ptr::null_mut(),
                                     0,
                                     0,
                                     ::std::ptr::null_mut());
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// a wrapper of PyArray_ZEROS
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ZEROS
    pub fn zeros(py: Python,
                 np: &PyArrayModule,
                 dims: &[usize],
                 typenum: NPY_TYPES,
                 order: NPY_ORDER)
                 -> Self {
        let dims: Vec<npy_intp> = dims.iter().map(|d| *d as npy_intp).collect();
        unsafe {
            let descr = np.PyArray_DescrFromType(typenum as i32);
            let ptr = np.PyArray_Zeros(dims.len() as i32,
                                       dims.as_ptr() as *mut npy_intp,
                                       descr,
                                       order as i32);
            Self::from_owned_ptr(py, ptr)
        }
    }

    /// a wrapper of PyArray_Arange
    /// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Arange
    pub fn arange(py: Python,
                  np: &PyArrayModule,
                  start: f64,
                  stop: f64,
                  step: f64,
                  typenum: NPY_TYPES)
                  -> Self {
        unsafe {
            let ptr = np.PyArray_Arange(start, stop, step, typenum as i32);
            Self::from_owned_ptr(py, ptr)
        }
    }
}

impl ToPyObject for PyArray {
    type ObjectType = Self;

    fn to_py_object(&self, py: Python) -> Self {
        PyClone::clone_ref(self, py)
    }
}

impl PythonObject for PyArray {
    #[inline]
    fn as_object(&self) -> &PyObject {
        &self.0
    }

    #[inline]
    fn into_object(self) -> PyObject {
        self.0
    }

    #[inline]
    unsafe fn unchecked_downcast_from(obj: PyObject) -> Self {
        PyArray(obj)
    }

    #[inline]
    unsafe fn unchecked_downcast_borrow_from<'a>(obj: &'a PyObject) -> &'a Self {
        ::std::mem::transmute(obj)
    }
}

impl PythonObjectWithCheckedDowncast for PyArray {
    fn downcast_from<'p>(py: Python<'p>,
                         obj: PyObject)
                         -> Result<PyArray, PythonObjectDowncastError<'p>> {
        let np = PyArrayModule::import(py).unwrap();
        unsafe {
            if npyffi::PyArray_Check(&np, obj.as_ptr()) != 0 {
                Ok(PyArray(obj))
            } else {
                Err(PythonObjectDowncastError(py))
            }
        }
    }

    fn downcast_borrow_from<'a, 'p>(py: Python<'p>,
                                    obj: &'a PyObject)
                                    -> Result<&'a PyArray, PythonObjectDowncastError<'p>> {
        let np = PyArrayModule::import(py).unwrap();
        unsafe {
            if npyffi::PyArray_Check(&np, obj.as_ptr()) != 0 {
                Ok(::std::mem::transmute(obj))
            } else {
                Err(PythonObjectDowncastError(py))
            }
        }
    }
}

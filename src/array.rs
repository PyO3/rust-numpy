
use np_ffi;
use py_ffi;

use cpython::*;

pub struct PyArray(PyObject);

impl PyArray {
    pub fn as_ptr(&self) -> *mut np_ffi::PyArrayObject {
        self.0.as_ptr() as *mut np_ffi::PyArrayObject
    }

    pub fn steal_ptr(self) -> *mut np_ffi::PyArrayObject {
        self.0.steal_ptr() as *mut np_ffi::PyArrayObject
    }

    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut np_ffi::PyArrayObject) -> Self {
        let obj = PyObject::from_owned_ptr(py, ptr as *mut py_ffi::PyObject);
        PyArray(obj)
    }

    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut np_ffi::PyArrayObject) -> Self {
        let obj = PyObject::from_borrowed_ptr(py, ptr as *mut py_ffi::PyObject);
        PyArray(obj)
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
        unsafe {
            if np_ffi::PyArray_Check(obj.as_ptr()) != 0 {
                Ok(PyArray(obj))
            } else {
                Err(PythonObjectDowncastError(py))
            }
        }
    }

    fn downcast_borrow_from<'a, 'p>(py: Python<'p>,
                                    obj: &'a PyObject)
                                    -> Result<&'a PyArray, PythonObjectDowncastError<'p>> {
        unsafe {
            if np_ffi::PyArray_Check(obj.as_ptr()) != 0 {
                Ok(::std::mem::transmute(obj))
            } else {
                Err(PythonObjectDowncastError(py))
            }
        }
    }
}

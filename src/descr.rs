
use np_ffi;
use py_ffi;

use cpython::*;

pub struct PyArrayDescr(PyObject);

impl PyArrayDescr {
    pub fn as_ptr(&self) -> *mut np_ffi::PyArray_Descr {
        self.0.as_ptr() as *mut np_ffi::PyArray_Descr
    }

    pub fn steal_ptr(self) -> *mut np_ffi::PyArray_Descr {
        self.0.steal_ptr() as *mut np_ffi::PyArray_Descr
    }

    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut np_ffi::PyArray_Descr) -> Self {
        let obj = PyObject::from_owned_ptr(py, ptr as *mut py_ffi::PyObject);
        PyArrayDescr(obj)
    }

    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut np_ffi::PyArray_Descr) -> Self {
        let obj = PyObject::from_borrowed_ptr(py, ptr as *mut py_ffi::PyObject);
        PyArrayDescr(obj)
    }

    pub fn new(py: Python, typenum: i32) -> Self {
        unsafe {
            let ptr = np_ffi::PyArray_DescrFromType(typenum);
            Self::from_owned_ptr(py, ptr)
        }
    }
}

impl ToPyObject for PyArrayDescr {
    type ObjectType = Self;

    fn to_py_object(&self, py: Python) -> Self {
        PyClone::clone_ref(self, py)
    }
}

impl PythonObject for PyArrayDescr {
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
        PyArrayDescr(obj)
    }

    #[inline]
    unsafe fn unchecked_downcast_borrow_from<'a>(obj: &'a PyObject) -> &'a Self {
        ::std::mem::transmute(obj)
    }
}

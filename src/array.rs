
use std::ptr::null_mut;
use std::os::raw::c_void;

use npffi;
use pyffi;
use cpython::*;
use npffi::types::npy_intp;
use super::NPY_TYPES;

pub struct PyArray(PyObject);

impl PyArray {
    pub fn as_ptr(&self) -> *mut npffi::PyArrayObject {
        self.0.as_ptr() as *mut npffi::PyArrayObject
    }

    pub fn steal_ptr(self) -> *mut npffi::PyArrayObject {
        self.0.steal_ptr() as *mut npffi::PyArrayObject
    }

    pub unsafe fn from_owned_ptr(py: Python, ptr: *mut pyffi::PyObject) -> Self {
        let obj = PyObject::from_owned_ptr(py, ptr);
        PyArray(obj)
    }

    pub unsafe fn from_borrowed_ptr(py: Python, ptr: *mut pyffi::PyObject) -> Self {
        let obj = PyObject::from_borrowed_ptr(py, ptr);
        PyArray(obj)
    }

    pub fn zeros(py: Python, dims: &[usize], typenum: NPY_TYPES, is_fortran_order: i32) -> Self {
        let dims: Vec<npy_intp> = dims.iter().map(|d| *d as npy_intp).collect();
        unsafe {
            let descr = npffi::PyArray_DescrFromType(typenum as i32);
            let ptr = npffi::PyArray_Zeros(dims.len() as i32,
                                           dims.as_ptr() as *mut npy_intp,
                                           descr,
                                           is_fortran_order);
            Self::from_owned_ptr(py, ptr)
        }
    }

    pub fn new(py: Python, dims: &[usize], typenum: NPY_TYPES) -> Self {
        let dims: Vec<npy_intp> = dims.iter().map(|d| *d as npy_intp).collect();
        unsafe {
            let ptr = npffi::PyArray_New(npffi::ARRAY_TYPE::PyArray_Type.as_type_object(),
                                         dims.len() as i32,
                                         dims.as_ptr() as *mut npy_intp,
                                         typenum as i32,
                                         null_mut::<isize>(),
                                         null_mut::<c_void>(),
                                         0,
                                         0,
                                         null_mut::<pyffi::PyObject>());
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
        unsafe {
            if npffi::PyArray_Check(obj.as_ptr()) != 0 {
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
            if npffi::PyArray_Check(obj.as_ptr()) != 0 {
                Ok(::std::mem::transmute(obj))
            } else {
                Err(PythonObjectDowncastError(py))
            }
        }
    }
}

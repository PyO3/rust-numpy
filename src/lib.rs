
extern crate python3_sys as py_ffi;
extern crate numpy_sys as np_ffi;
extern crate cpython;

use cpython::PyObject;

pub struct PyArray(PyObject);

impl PyArray {
    pub fn as_ptr(&self) -> *mut np_ffi::PyArrayObject {
        self.0.as_ptr() as *mut np_ffi::PyArrayObject
    }
    pub fn steal_ptr(self) -> *mut np_ffi::PyArrayObject {
        self.0.steal_ptr() as *mut np_ffi::PyArrayObject
    }
}

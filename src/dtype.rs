use crate::npyffi::{NpyTypes, PyArray_Descr, PY_ARRAY_API};
use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::c_int;

pub struct PyArrayDescr(PyAny);

pyobject_native_type_core!(
    PyArrayDescr,
    PyArray_Descr,
    *PY_ARRAY_API.get_type_object(NpyTypes::PyArrayDescr_Type),
    Some("numpy"),
    arraydescr_check
);

pyobject_native_type_fmt!(PyArrayDescr);

unsafe fn arraydescr_check(op: *mut ffi::PyObject) -> c_int {
    ffi::PyObject_TypeCheck(
        op,
        PY_ARRAY_API.get_type_object(NpyTypes::PyArrayDescr_Type),
    )
}

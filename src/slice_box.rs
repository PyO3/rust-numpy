use pyo3::class::methods::{PyMethodDefType, PyMethodsProtocol};
use pyo3::{ffi, type_object, types::PyAny, AsPyPointer, PyObjectAlloc, Python};
use std::os::raw::c_void;

/// It's a memory store for IntoPyArray.
/// See IntoPyArray's doc for what concretely this type is for.
#[repr(C)]
pub(crate) struct SliceBox<T> {
    ob_base: ffi::PyObject,
    inner: *mut [T],
}

impl<T> SliceBox<T> {
    pub(crate) unsafe fn new<'a>(box_: Box<[T]>) -> &'a Self {
        let type_ob = <Self as type_object::PyTypeObject>::init_type().as_ptr();
        let base = ffi::_PyObject_New(type_ob);
        *base = ffi::PyObject_HEAD_INIT;
        (*base).ob_type = type_ob;
        let self_ = base as *mut SliceBox<T>;
        (*self_).inner = Box::into_raw(box_);
        &*self_
    }
    pub(crate) fn data(&self) -> *mut c_void {
        self.inner as *mut c_void
    }
}

impl<T> type_object::PyTypeInfo for SliceBox<T> {
    type Type = ();
    type BaseType = PyAny;
    const NAME: &'static str = "SliceBox";
    const DESCRIPTION: &'static str = "Memory store for PyArray using rust's Box<[T]>.";
    const FLAGS: usize = 0;
    const SIZE: usize = std::mem::size_of::<Self>();
    const OFFSET: isize = 0;
    #[inline]
    unsafe fn type_object() -> &'static mut ffi::PyTypeObject {
        static mut TYPE_OBJECT: ::pyo3::ffi::PyTypeObject = ::pyo3::ffi::PyTypeObject_INIT;
        &mut TYPE_OBJECT
    }
}

impl<T> PyMethodsProtocol for SliceBox<T> {
    fn py_methods() -> Vec<&'static PyMethodDefType> {
        Vec::new()
    }
}

impl<T> AsPyPointer for SliceBox<T> {
    #[inline]
    fn as_ptr(&self) -> *mut ffi::PyObject {
        &self.ob_base as *const _ as *mut _
    }
}

impl<T> PyObjectAlloc for SliceBox<T> {
    /// Calls the rust destructor for the object.
    unsafe fn drop(py: Python<'_>, obj: *mut ffi::PyObject) {
        let data = (*(obj as *mut SliceBox<T>)).inner;
        let boxed_slice = Box::from_raw(data);
        drop(boxed_slice);
        <Self as type_object::PyTypeInfo>::BaseType::drop(py, obj);
    }
    unsafe fn dealloc(py: Python<'_>, obj: *mut ffi::PyObject) {
        Self::drop(py, obj);
        ffi::PyObject_Free(obj as *mut c_void);
    }
}

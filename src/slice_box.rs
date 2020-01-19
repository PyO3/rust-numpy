use pyo3::class::methods::{PyMethodDefType, PyMethodsProtocol};
use pyo3::pyclass::{PyClass, PyClassAlloc, PyClassShell};
use pyo3::pyclass_slots::PyClassDummySlot;
use pyo3::{ffi, type_object, types::PyAny, PyClassInitializer, PyResult, Python};

pub(crate) struct SliceBox<T> {
    pub(crate) data: *mut [T],
}

impl<T> SliceBox<T> {
    pub(crate) unsafe fn new(
        py: Python<'_>,
        value: Box<[T]>,
    ) -> PyResult<*mut PyClassShell<SliceBox<T>>> {
        let value = SliceBox {
            data: Box::into_raw(value),
        };
        PyClassInitializer::from(value).create_shell(py)
    }
}

impl<T> Drop for SliceBox<T> {
    fn drop(&mut self) {
        let _boxed_slice = unsafe { Box::from_raw(self.data) };
    }
}

impl<T> PyClassAlloc for SliceBox<T> {}

impl<T> PyClass for SliceBox<T> {
    type Dict = PyClassDummySlot;
    type WeakRef = PyClassDummySlot;
}

impl<T> type_object::PyTypeInfo for SliceBox<T> {
    type Type = ();
    type BaseType = PyAny;
    type ConcreteLayout = PyClassShell<Self>;
    type Initializer = PyClassInitializer<Self>;
    const NAME: &'static str = "SliceBox";
    const MODULE: Option<&'static str> = Some("_rust_numpy");
    const DESCRIPTION: &'static str = "Memory store for PyArray using rust's Box<[T]>.";
    const FLAGS: usize = 0;

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

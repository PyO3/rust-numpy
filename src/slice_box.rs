use pyo3::class::{methods::PyMethods, proto_methods::PyProtoMethods};
use pyo3::pyclass::{PyClass, PyClassAlloc, PyClassSend, ThreadCheckerStub};
use pyo3::pyclass_slots::PyClassDummySlot;
use pyo3::{ffi, type_object, types::PyAny, PyCell, PyClassInitializer};

pub(crate) struct SliceBox<T> {
    pub(crate) data: *mut [T],
}

impl<T> SliceBox<T> {
    pub(crate) fn new(value: Box<[T]>) -> Self {
        SliceBox {
            data: Box::into_raw(value),
        }
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
    type BaseNativeType = PyAny;
}

unsafe impl<T> type_object::PyTypeInfo for SliceBox<T> {
    type Type = ();
    type BaseType = PyAny;
    type BaseLayout = pyo3::pycell::PyCellBase<PyAny>;
    type Layout = PyCell<Self>;
    type Initializer = PyClassInitializer<Self>;
    type AsRefTarget = PyCell<Self>;
    const NAME: &'static str = "SliceBox";
    const MODULE: Option<&'static str> = Some("_rust_numpy");
    const DESCRIPTION: &'static str = "Memory store for PyArray using rust's Box<[T]> \0";
    const FLAGS: usize = 0;

    #[inline]
    fn type_object_raw(py: pyo3::Python) -> *mut ffi::PyTypeObject {
        use pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}

// Some stubs to use PyClass
impl<T> PyMethods for SliceBox<T> {}
impl<T> PyProtoMethods for SliceBox<T> {}
unsafe impl<T> Send for SliceBox<T> {}
impl<T> PyClassSend for SliceBox<T> {
    type ThreadChecker = ThreadCheckerStub<Self>;
}

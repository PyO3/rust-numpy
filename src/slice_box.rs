use pyo3::class::impl_::{PyClassImpl, ThreadCheckerStub};
use pyo3::pyclass::PyClass;
use pyo3::pyclass_slots::PyClassDummySlot;
use pyo3::type_object::{LazyStaticType, PyTypeInfo};
use pyo3::{ffi, types::PyAny, PyCell};

pub(crate) struct SliceBox {
    ptr: *mut [u8],
    drop: unsafe fn(*mut [u8]),
}

unsafe impl Send for SliceBox {}

impl SliceBox {
    pub(crate) fn new<T: Send>(data: Box<[T]>) -> Self {
        unsafe fn drop_boxed_slice<T>(ptr: *mut [u8]) {
            let _ = Box::from_raw(ptr as *mut [T]);
        }

        let ptr = Box::into_raw(data) as *mut [u8];
        let drop = drop_boxed_slice::<T>;

        Self { ptr, drop }
    }
}

impl Drop for SliceBox {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr);
        }
    }
}

impl PyClass for SliceBox {
    type Dict = PyClassDummySlot;
    type WeakRef = PyClassDummySlot;
    type BaseNativeType = PyAny;
}

impl PyClassImpl for SliceBox {
    const DOC: &'static str = "Memory store for PyArray using rust's Box<[T]> \0";

    type BaseType = PyAny;
    type Layout = PyCell<Self>;
    type ThreadChecker = ThreadCheckerStub<Self>;
}

unsafe impl PyTypeInfo for SliceBox {
    type AsRefTarget = PyCell<Self>;
    const NAME: &'static str = "SliceBox";
    const MODULE: Option<&'static str> = Some("_rust_numpy");

    #[inline]
    fn type_object_raw(py: pyo3::Python) -> *mut ffi::PyTypeObject {
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}

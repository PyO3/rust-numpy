use std::{mem, slice};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use pyo3::{
    ffi,
    impl_::pyclass::{PyClassDummySlot, PyClassImpl, PyClassItems, ThreadCheckerStub},
    pyclass::PyClass,
    type_object::{LazyStaticType, PyTypeInfo},
    PyAny, PyCell, Python,
};

/// Utility type to safely store `Box<[_]>` or `Vec<_>` on the Python heap
pub(crate) struct PySliceContainer {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    drop: unsafe fn(*mut u8, usize, usize),
}

unsafe impl Send for PySliceContainer {}

impl<T: Send> From<Box<[T]>> for PySliceContainer {
    fn from(data: Box<[T]>) -> Self {
        unsafe fn drop_boxed_slice<T>(ptr: *mut u8, len: usize, _cap: usize) {
            let _ = Box::from_raw(slice::from_raw_parts_mut(ptr as *mut T, len) as *mut [T]);
        }

        // FIXME(adamreichold): Use `Box::into_raw` when
        // `*mut [T]::{as_mut_ptr, len}` become stable and compatible with our MSRV.
        let ptr = data.as_ptr() as *mut u8;
        let len = data.len();
        let cap = 0;
        let drop = drop_boxed_slice::<T>;

        mem::forget(data);

        Self {
            ptr,
            len,
            cap,
            drop,
        }
    }
}

impl<T: Send> From<Vec<T>> for PySliceContainer {
    fn from(data: Vec<T>) -> Self {
        unsafe fn drop_vec<T>(ptr: *mut u8, len: usize, cap: usize) {
            let _ = Vec::from_raw_parts(ptr as *mut T, len, cap);
        }

        // FIXME(adamreichold): Use `Vec::into_raw_parts`
        // when it becomes stable and compatible with our MSRV.
        let ptr = data.as_ptr() as *mut u8;
        let len = data.len();
        let cap = data.capacity();
        let drop = drop_vec::<T>;

        mem::forget(data);

        Self {
            ptr,
            len,
            cap,
            drop,
        }
    }
}

impl<A, D> From<ArrayBase<OwnedRepr<A>, D>> for PySliceContainer
where
    A: Send,
    D: Dimension,
{
    fn from(data: ArrayBase<OwnedRepr<A>, D>) -> Self {
        Self::from(data.into_raw_vec())
    }
}

impl Drop for PySliceContainer {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr, self.len, self.cap);
        }
    }
}

impl PyClass for PySliceContainer {
    type Dict = PyClassDummySlot;
    type WeakRef = PyClassDummySlot;
    type BaseNativeType = PyAny;
}

impl PyClassImpl for PySliceContainer {
    const DOC: &'static str = "Memory store for a PyArray backed by a Box<[_]> or a Vec<_> \0";

    type BaseType = PyAny;
    type Layout = PyCell<Self>;
    type ThreadChecker = ThreadCheckerStub<Self>;

    fn for_all_items(_visitor: &mut dyn FnMut(&PyClassItems)) {}
}

unsafe impl PyTypeInfo for PySliceContainer {
    type AsRefTarget = PyCell<Self>;

    const NAME: &'static str = "PySliceContainer";
    const MODULE: Option<&'static str> = Some("_rust_numpy");

    #[inline]
    fn type_object_raw(py: Python) -> *mut ffi::PyTypeObject {
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}

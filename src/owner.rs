use std::{mem, slice};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use pyo3::class::impl_::{PyClassImpl, ThreadCheckerStub};
use pyo3::pyclass::PyClass;
use pyo3::pyclass_slots::PyClassDummySlot;
use pyo3::type_object::{LazyStaticType, PyTypeInfo};
use pyo3::{ffi, types::PyAny, PyCell};

use crate::dtype::Element;

pub(crate) struct Owner {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    drop: unsafe fn(*mut u8, usize, usize),
}

unsafe impl Send for Owner {}

impl<T: Send> From<Box<[T]>> for Owner {
    fn from(data: Box<[T]>) -> Self {
        unsafe fn drop_boxed_slice<T>(ptr: *mut u8, len: usize, _cap: usize) {
            let _ = Box::from_raw(slice::from_raw_parts_mut(ptr as *mut T, len) as *mut [T]);
        }

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

impl<T: Send> From<Vec<T>> for Owner {
    fn from(data: Vec<T>) -> Self {
        unsafe fn drop_vec<T>(ptr: *mut u8, len: usize, cap: usize) {
            let _ = Vec::from_raw_parts(ptr as *mut T, len, cap);
        }

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

impl<A, D> From<ArrayBase<OwnedRepr<A>, D>> for Owner
where
    A: Element,
    D: Dimension,
{
    fn from(data: ArrayBase<OwnedRepr<A>, D>) -> Self {
        Self::from(data.into_raw_vec())
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr, self.len, self.cap);
        }
    }
}

impl PyClass for Owner {
    type Dict = PyClassDummySlot;
    type WeakRef = PyClassDummySlot;
    type BaseNativeType = PyAny;
}

impl PyClassImpl for Owner {
    const DOC: &'static str = "Memory store for a PyArray backed by a Rust type \0";

    type BaseType = PyAny;
    type Layout = PyCell<Self>;
    type ThreadChecker = ThreadCheckerStub<Self>;
}

unsafe impl PyTypeInfo for Owner {
    type AsRefTarget = PyCell<Self>;

    const NAME: &'static str = "Owner";
    const MODULE: Option<&'static str> = Some("_rust_numpy");

    #[inline]
    fn type_object_raw(py: pyo3::Python) -> *mut ffi::PyTypeObject {
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}

use crate::types::TypeNum;
use pyo3::{self, ffi, typeob, PyObjectAlloc, Python, ToPyPointer};
use std::os::raw::c_void;

#[repr(C)]
pub(crate) struct SliceBox<T> {
    ob_base: ffi::PyObject,
    inner: *mut [T],
}

impl<T> SliceBox<T> {
    pub(crate) unsafe fn new<'a>(box_: Box<[T]>) -> &'a Self {
        <Self as typeob::PyTypeObject>::init_type();
        let type_ob = <Self as typeob::PyTypeInfo>::type_object() as *mut _;
        let base = ffi::_PyObject_New(type_ob);
        *base = ffi::PyObject_HEAD_INIT;
        (*base).ob_type = <Self as typeob::PyTypeInfo>::type_object() as *mut _;
        let self_ = base as *mut SliceBox<T>;
        (*self_).inner = Box::into_raw(box_);
        &*self_
    }
    pub(crate) fn data(&self) -> *mut c_void {
        self.inner as *mut c_void
    }
}

impl<T> typeob::PyTypeInfo for SliceBox<T> {
    type Type = ();
    type BaseType = pyo3::PyObjectRef;
    const NAME: &'static str = "SliceBox";
    const DESCRIPTION: &'static str = "Memory store for PyArray made by IntoPyArray.";
    const FLAGS: usize = 0;
    const SIZE: usize = { Self::OFFSET as usize + std::mem::size_of::<Self>() + 0 + 0 };
    const OFFSET: isize = 0;
    #[inline]
    unsafe fn type_object() -> &'static mut ::pyo3::ffi::PyTypeObject {
        static mut TYPE_OBJECT: ::pyo3::ffi::PyTypeObject = ::pyo3::ffi::PyTypeObject_INIT;
        &mut TYPE_OBJECT
    }
}

impl<T: TypeNum> typeob::PyTypeObject for SliceBox<T> {
    #[inline(always)]
    fn init_type() {
        static START: std::sync::Once = std::sync::ONCE_INIT;
        START.call_once(|| {
            let ty = unsafe { <Self as typeob::PyTypeInfo>::type_object() };
            if (ty.tp_flags & ffi::Py_TPFLAGS_READY) == 0 {
                let gil = Python::acquire_gil();
                let py = gil.python();
                let mod_name = format!("rust_numpy.{:?}", T::npy_data_type());
                typeob::initialize_type::<Self>(py, Some(&mod_name))
                    .map_err(|e| e.print(py))
                    .expect("Failed to initialize SliceBox");
            }
        });
    }
}

impl<T> ToPyPointer for SliceBox<T> {
    #[inline]
    fn as_ptr(&self) -> *mut ffi::PyObject {
        &self.ob_base as *const _ as *mut _
    }
}

impl<T> PyObjectAlloc<SliceBox<T>> for SliceBox<T> {
    /// Calls the rust destructor for the object.
    unsafe fn drop(py: Python, obj: *mut ffi::PyObject) {
        let data = (*(obj as *mut SliceBox<T>)).inner;
        let box_ = Box::from_raw(data);
        drop(box_);
        <Self as typeob::PyTypeInfo>::BaseType::drop(py, obj);
    }
    unsafe fn dealloc(py: Python, obj: *mut ffi::PyObject) {
        Self::drop(py, obj);
        ffi::PyObject_Free(obj as *mut c_void);
    }
}


use cpython::Python;
use std::os::raw::c_void;

use super::*;

pub trait IntoPyArray {
    fn into_pyarray(self, Python, &PyArrayModule) -> PyArray;
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray {
        let mut dims = [self.len() as npy_intp];
        unsafe {
            let ptr: *mut [T] = Box::into_raw(self.into_boxed_slice());
            let data = (*ptr).as_ref().as_ptr() as *mut c_void;
            let ptr = np.PyArray_New(np.get_type_object(super::npyffi::ArrayType::PyArray_Type),
                                     dims.len() as i32,
                                     dims.as_mut_ptr(),
                                     T::typenum(),
                                     ::std::ptr::null_mut(),
                                     data,
                                     0,
                                     0,
                                     ::std::ptr::null_mut());
            PyArray::from_owned_ptr(py, ptr)
        }
    }
}

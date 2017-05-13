
use cpython::Python;
use std::os::raw::c_void;
use std::ptr::null_mut;
use std::iter::Iterator;

use super::*;

pub trait IntoPyArray {
    fn into_pyarray(self, Python, &PyArrayModule) -> PyArray;
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray {
        let dims = [self.len()];
        unsafe {
            let ptr: *mut [T] = Box::into_raw(self.into_boxed_slice());
            let data = (*ptr).as_ref().as_ptr() as *mut c_void;
            PyArray::new_::<T>(py, np, &dims, null_mut(), data)
        }
    }
}

pub trait ToPyArray {
    fn to_pyarray(self, Python, &PyArrayModule) -> PyArray;
}

impl<Iter, T: TypeNum> ToPyArray for Iter
    where Iter: Iterator<Item = T> + Sized
{
    fn to_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray {
        let vec: Vec<T> = self.collect();
        vec.into_pyarray(py, np)
    }
}


use cpython::Python;
use ndarray::*;

use std::os::raw::c_void;
use std::ptr::null_mut;
use std::iter::Iterator;
use std::mem::size_of;

use super::*;

pub trait IntoPyArray {
    fn into_pyarray(self, Python, &PyArrayModule) -> PyArray;
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray {
        let dims = [self.len()];
        unsafe { PyArray::new_::<T>(py, np, &dims, null_mut(), into_raw(self)) }
    }
}

impl<A: TypeNum, D: Dimension> IntoPyArray for Array<A, D> {
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray {
        let dims: Vec<usize> = self.shape().iter().cloned().collect();
        let mut strides: Vec<npy_intp> = self.strides()
            .into_iter()
            .map(|n| n * size_of::<A>() as npy_intp)
            .collect();
        unsafe {
            let data = into_raw(self.into_raw_vec());
            PyArray::new_::<A>(py, np, &dims, strides.as_mut_ptr(), data)
        }
    }
}

unsafe fn into_raw<T>(x: Vec<T>) -> *mut c_void {
    let ptr: *mut [T] = Box::into_raw(x.into_boxed_slice());
    (*ptr).as_ref().as_ptr() as *mut c_void
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

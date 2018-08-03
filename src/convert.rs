use ndarray::*;
use pyo3::Python;

use std::iter::Iterator;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::null_mut;

use super::*;

pub trait IntoPyArray {
    type Item: TypeNum;
    fn into_pyarray(self, Python, &PyArrayModule) -> PyArray<Self::Item>;
}

impl<T: TypeNum> IntoPyArray for Box<[T]> {
    type Item = T;
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray<Self::Item> {
        let dims = [self.len()];
        let ptr = Box::into_raw(self);
        unsafe { PyArray::new_(py, np, &dims, null_mut(), ptr as *mut c_void) }
    }
}

impl<T: TypeNum> IntoPyArray for Vec<T> {
    type Item = T;
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray<Self::Item> {
        let dims = [self.len()];
        unsafe { PyArray::new_(py, np, &dims, null_mut(), into_raw(self)) }
    }
}

impl<A: TypeNum, D: Dimension> IntoPyArray for Array<A, D> {
    type Item = A;
    fn into_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray<Self::Item> {
        let dims: Vec<_> = self.shape().iter().cloned().collect();
        let mut strides: Vec<_> = self
            .strides()
            .into_iter()
            .map(|n| n * size_of::<A>() as npy_intp)
            .collect();
        unsafe {
            let data = into_raw(self.into_raw_vec());
            PyArray::new_(py, np, &dims, strides.as_mut_ptr(), data)
        }
    }
}

macro_rules! array_impls {
    ($($N: expr)+) => {
        $(
            impl<T: TypeNum> IntoPyArray for [T; $N] {
                type Item = T;
                fn into_pyarray(mut self, py: Python, np: &PyArrayModule) -> PyArray<T> {
                    let dims = [$N];
                    let ptr = &mut self as *mut [T; $N];
                    unsafe {
                        PyArray::new_(py, np, &dims, null_mut(), ptr as *mut c_void)
                    }
                }
            }
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}

pub(crate) unsafe fn into_raw<T>(x: Vec<T>) -> *mut c_void {
    let ptr = Box::into_raw(x.into_boxed_slice());
    ptr as *mut c_void
}

pub trait ToPyArray {
    type Item: TypeNum;
    fn to_pyarray(self, Python, &PyArrayModule) -> PyArray<Self::Item>;
}

impl<Iter, T: TypeNum> ToPyArray for Iter
where
    Iter: Iterator<Item = T> + Sized,
{
    type Item = T;
    fn to_pyarray(self, py: Python, np: &PyArrayModule) -> PyArray<Self::Item> {
        let vec: Vec<T> = self.collect();
        vec.into_pyarray(py, np)
    }
}

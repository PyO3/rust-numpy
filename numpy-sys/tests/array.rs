
extern crate python3_sys as pyffi;
extern crate numpy_sys;

use pyffi::*;
use numpy_sys::*;
use std::os::raw::c_void;

unsafe fn get_api(offset: isize) -> *const *const c_void {
    let api = &PyArray_API as *const *const c_void;
    api.offset(offset)
}

#[test]
fn test_no_pyobject() {
    let res = unsafe {
        let fptr = get_api(0) as (*const extern "C" fn() -> u32);
        (*fptr)()
    };
    println!("ndarray C version = {}", res);
}

#[test]
fn test_return_pyobject() {
    let _res = unsafe {
        let fptr = get_api(186) as (*const extern "C" fn(f64, f64, f64, i32) -> *mut PyObject);
        (*fptr)(0.0, 1.0, 0.1, 12)
    };
}

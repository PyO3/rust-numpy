
extern crate python3_sys as pyffi;
extern crate numpy_sys;

use numpy_sys::*;
use std::os::raw::c_void;

unsafe fn get_api(offset: isize) -> *const *const c_void {
    let api = &PyArray_API as *const *const c_void;
    api.offset(offset)
}

/// check link succeeds
#[test]
fn test_link() {
    let res = unsafe {
        let fptr = get_api(0) as (*const extern "C" fn() -> u32);
        (*fptr)()
    };
    println!("ndarray C version = {}", res);
}


extern crate numpy_sys;

use numpy_sys::*;

#[test]
fn link_multiarray() {
    let res = unsafe { PyArray_GetNDArrayCVersion() };
    println!("ndarray C version = {}", res);
}

#[test]
fn link_umath() {
    unsafe { PyUFunc_clearfperr() };
}

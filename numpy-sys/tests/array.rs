
extern crate numpy_sys;

use numpy_sys::*;

#[test]
fn test_link() {
    // check link succeeds
    let _ara = unsafe { *PyArray_API };
}

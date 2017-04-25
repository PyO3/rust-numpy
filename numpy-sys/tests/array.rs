
extern crate numpy_sys;

use numpy_sys::*;

#[test]
fn test_link() {
    // check link succeeds
    let _arange = unsafe { PyArray_Arange(0.0, 1.0, 0.1, 0) };
}

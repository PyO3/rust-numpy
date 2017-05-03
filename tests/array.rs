
extern crate cpython;
extern crate numpy;

use numpy::*;

#[test]
fn array_shapes() {
    let py = cpython::Python::acquire_gil();
    let n = 3;
    let m = 5;
    let arr = PyArray::zeros(py.python(),
                             &[n, m],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_CORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [m as isize * 8, 8]);

    let arr = PyArray::zeros(py.python(),
                             &[n, m],
                             NPY_TYPES::NPY_DOUBLE,
                             NPY_ORDER::NPY_FORTRANORDER);
    assert!(arr.ndim() == 2);
    assert!(arr.dims() == [n, m]);
    assert!(arr.strides() == [8, n as isize * 8]);
}

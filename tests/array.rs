
extern crate cpython;
extern crate numpy;

#[test]
fn array_new() {
    let py = cpython::Python::acquire_gil();
    let _arr = numpy::PyArray::new(py.python(), &[4, 4], numpy::NPY_TYPES::NPY_DOUBLE);
}

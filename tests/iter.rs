use ndarray::array;
use numpy::{NpyMultiIterBuilder, NpySingleIterBuilder, PyArray};
use pyo3::PyResult;

#[test]
fn get_iter() {
    let gil = pyo3::Python::acquire_gil();
    let dim = (3, 5);
    let arr = PyArray::<f64, _>::zeros(gil.python(), dim, false);
    let mut iter = NpySingleIterBuilder::readonly(arr)
        .build()
        .map_err(|e| e.print(gil.python()))
        .unwrap();
    assert_eq!(*iter.next().unwrap(), 0.0);
}

#[test]
fn sum_iter() -> PyResult<()> {
    let gil = pyo3::Python::acquire_gil();
    let data = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];

    let arr = PyArray::from_array(gil.python(), &data);
    let iter = NpySingleIterBuilder::readonly(arr)
        .build()
        .map_err(|e| e.print(gil.python()))
        .unwrap();

    // The order of iteration is not specified, so we should restrict ourselves
    // to tests that don't verify a given order.
    assert_eq!(iter.sum::<f64>(), 15.0);
    Ok(())
}

#[test]
fn multi_iter() -> PyResult<()> {
    let gil = pyo3::Python::acquire_gil();
    let data1 = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let data2 = array![[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]];

    let py = gil.python();
    let arr1 = PyArray::from_array(py, &data1);
    let arr2 = PyArray::from_array(py, &data2);
    let iter = NpyMultiIterBuilder::new()
        .add_readonly_array(arr1)
        .add_readonly_array(arr2)
        .build()
        .map_err(|e| e.print(py))
        .unwrap();

    let mut sum = 0.0;
    for (x, y) in iter {
        sum += *x * *y;
    }

    assert_eq!(sum, 145.0);
    Ok(())
}

use numpy::{npyiter::NpyIterFlag, *};
use pyo3::PyResult;

#[test]
fn get_iter() {
    let gil = pyo3::Python::acquire_gil();
    let dim = (3, 5);
    let arr = PyArray::<f64, _>::zeros(gil.python(), dim, false);
    let mut iter = npyiter::NpyIterBuilder::new(arr)
        .add(NpyIterFlag::ReadOnly)
        .build()
        .map_err(|e| e.print(gil.python()))
        .unwrap();
    assert_eq!(*iter.next().unwrap(), 0.0);
}

#[test]
fn sum_iter() -> PyResult<()> {
    let gil = pyo3::Python::acquire_gil();
    let vec_data = vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![4.0, 5.0]];

    let arr = PyArray::from_vec2(gil.python(), &vec_data)?;
    let iter = npyiter::NpyIterBuilder::new(arr)
        .add(NpyIterFlag::ReadOnly)
        .build()
        .map_err(|e| e.print(gil.python()))
        .unwrap();

    // The order of iteration is not specified, so we should restrict ourselves
    // to tests that don't verify a given order.
    assert_eq!(iter.sum::<f64>(), 15.0);
    Ok(())
}

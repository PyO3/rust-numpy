use numpy::{npyiter::NpyIterFlag, *};

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

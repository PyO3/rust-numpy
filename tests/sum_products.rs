use numpy::prelude::*;
use numpy::{
    array, dot_bound, einsum_bound, inner_bound, pyarray_bound, PyArray0, PyArray1, PyArray2,
};
use pyo3::{Bound, Python};

#[test]
fn test_dot() {
    Python::with_gil(|py| {
        let a = pyarray_bound![py, [1, 0], [0, 1]];
        let b = pyarray_bound![py, [4, 1], [2, 2]];
        let c: Bound<'_, PyArray2<_>> = dot_bound(&a, &b).unwrap();
        assert_eq!(c.readonly().as_array(), array![[4, 1], [2, 2]]);

        let a = pyarray_bound![py, 1, 2, 3];
        let err = dot_bound::<_, _, _, Bound<'_, PyArray2<_>>>(&a, &b).unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{}", err);

        let a = pyarray_bound![py, 1, 2, 3];
        let b = pyarray_bound![py, 0, 1, 0];
        let c: Bound<'_, PyArray0<_>> = dot_bound(&a, &b).unwrap();
        assert_eq!(c.item(), 2);
        let c: i32 = dot_bound(&a, &b).unwrap();
        assert_eq!(c, 2);

        let a = pyarray_bound![py, 1.0, 2.0, 3.0];
        let b = pyarray_bound![py, 0.0, 0.0, 0.0];
        let c: f64 = dot_bound(&a, &b).unwrap();
        assert_eq!(c, 0.0);
    });
}

#[test]
fn test_inner() {
    Python::with_gil(|py| {
        let a = pyarray_bound![py, 1, 2, 3];
        let b = pyarray_bound![py, 0, 1, 0];
        let c: Bound<'_, PyArray0<_>> = inner_bound(&a, &b).unwrap();
        assert_eq!(c.item(), 2);
        let c: i32 = inner_bound(&a, &b).unwrap();
        assert_eq!(c, 2);

        let a = pyarray_bound![py, 1.0, 2.0, 3.0];
        let b = pyarray_bound![py, 0.0, 0.0, 0.0];
        let c: f64 = inner_bound(&a, &b).unwrap();
        assert_eq!(c, 0.0);

        let a = pyarray_bound![py, [1, 0], [0, 1]];
        let b = pyarray_bound![py, [4, 1], [2, 2]];
        let c: Bound<'_, PyArray2<_>> = inner_bound(&a, &b).unwrap();
        assert_eq!(c.readonly().as_array(), array![[4, 2], [1, 2]]);

        let a = pyarray_bound![py, 1, 2, 3];
        let err = inner_bound::<_, _, _, Bound<'_, PyArray2<_>>>(&a, &b).unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{}", err);
    });
}

#[test]
fn test_einsum() {
    Python::with_gil(|py| {
        let a = PyArray1::<i32>::arange_bound(py, 0, 25, 1)
            .reshape([5, 5])
            .unwrap();
        let b = pyarray_bound![py, 0, 1, 2, 3, 4];
        let c = pyarray_bound![py, [0, 1, 2], [3, 4, 5]];

        let d: Bound<'_, PyArray0<_>> = einsum_bound!("ii", a).unwrap();
        assert_eq!(d.item(), 60);

        let d: i32 = einsum_bound!("ii", a).unwrap();
        assert_eq!(d, 60);

        let d: Bound<'_, PyArray1<_>> = einsum_bound!("ii->i", a).unwrap();
        assert_eq!(d.readonly().as_array(), array![0, 6, 12, 18, 24]);

        let d: Bound<'_, PyArray1<_>> = einsum_bound!("ij->i", a).unwrap();
        assert_eq!(d.readonly().as_array(), array![10, 35, 60, 85, 110]);

        let d: Bound<'_, PyArray2<_>> = einsum_bound!("ji", c).unwrap();
        assert_eq!(d.readonly().as_array(), array![[0, 3], [1, 4], [2, 5]]);

        let d: Bound<'_, PyArray1<_>> = einsum_bound!("ij,j", a, b).unwrap();
        assert_eq!(d.readonly().as_array(), array![30, 80, 130, 180, 230]);
    });
}

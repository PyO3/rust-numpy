use numpy::prelude::*;
use numpy::{array, dot, einsum, inner, pyarray, PyArray0, PyArray1, PyArray2};
use pyo3::{Bound, Python};

#[test]
fn test_dot() {
    Python::attach(|py| {
        let a = pyarray![py, [1, 0], [0, 1]];
        let b = pyarray![py, [4, 1], [2, 2]];
        let c: Bound<'_, PyArray2<_>> = dot(&a, &b).unwrap();
        assert_eq!(c.readonly().as_array(), array![[4, 1], [2, 2]]);

        let a = pyarray![py, 1, 2, 3];
        let err = dot::<_, _, _, Bound<'_, PyArray2<_>>>(&a, &b).unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{}", err);

        let a = pyarray![py, 1, 2, 3];
        let b = pyarray![py, 0, 1, 0];
        let c: Bound<'_, PyArray0<_>> = dot(&a, &b).unwrap();
        assert_eq!(c.item(), 2);
        let c: i32 = dot(&a, &b).unwrap();
        assert_eq!(c, 2);

        let a = pyarray![py, 1.0, 2.0, 3.0];
        let b = pyarray![py, 0.0, 0.0, 0.0];
        let c: f64 = dot(&a, &b).unwrap();
        assert_eq!(c, 0.0);
    });
}

#[test]
fn test_inner() {
    Python::attach(|py| {
        let a = pyarray![py, 1, 2, 3];
        let b = pyarray![py, 0, 1, 0];
        let c: Bound<'_, PyArray0<_>> = inner(&a, &b).unwrap();
        assert_eq!(c.item(), 2);
        let c: i32 = inner(&a, &b).unwrap();
        assert_eq!(c, 2);

        let a = pyarray![py, 1.0, 2.0, 3.0];
        let b = pyarray![py, 0.0, 0.0, 0.0];
        let c: f64 = inner(&a, &b).unwrap();
        assert_eq!(c, 0.0);

        let a = pyarray![py, [1, 0], [0, 1]];
        let b = pyarray![py, [4, 1], [2, 2]];
        let c: Bound<'_, PyArray2<_>> = inner(&a, &b).unwrap();
        assert_eq!(c.readonly().as_array(), array![[4, 2], [1, 2]]);

        let a = pyarray![py, 1, 2, 3];
        let err = inner::<_, _, _, Bound<'_, PyArray2<_>>>(&a, &b).unwrap_err();
        assert!(err.to_string().contains("not aligned"), "{}", err);
    });
}

#[test]
fn test_einsum() {
    Python::attach(|py| {
        let a = PyArray1::<i32>::arange(py, 0, 25, 1)
            .reshape([5, 5])
            .unwrap();
        let b = pyarray![py, 0, 1, 2, 3, 4];
        let c = pyarray![py, [0, 1, 2], [3, 4, 5]];

        let d: Bound<'_, PyArray0<_>> = einsum!("ii", a).unwrap();
        assert_eq!(d.item(), 60);

        let d: i32 = einsum!("ii", a).unwrap();
        assert_eq!(d, 60);

        let d: Bound<'_, PyArray1<_>> = einsum!("ii->i", a).unwrap();
        assert_eq!(d.readonly().as_array(), array![0, 6, 12, 18, 24]);

        let d: Bound<'_, PyArray1<_>> = einsum!("ij->i", a).unwrap();
        assert_eq!(d.readonly().as_array(), array![10, 35, 60, 85, 110]);

        let d: Bound<'_, PyArray2<_>> = einsum!("ji", c).unwrap();
        assert_eq!(d.readonly().as_array(), array![[0, 3], [1, 4], [2, 5]]);

        let d: Bound<'_, PyArray1<_>> = einsum!("ij,j", a, b).unwrap();
        assert_eq!(d.readonly().as_array(), array![30, 80, 130, 180, 230]);
    });
}

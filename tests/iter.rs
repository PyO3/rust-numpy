#![allow(deprecated)]

use ndarray::array;
use numpy::{pyarray, NpyMultiIterBuilder, NpySingleIterBuilder, PyArray};
use pyo3::{PyResult, Python};

macro_rules! assert_approx_eq {
    ($x: expr, $y: expr) => {
        assert!(($x - $y) <= std::f64::EPSILON);
    };
}

#[test]
fn readonly_iter() -> PyResult<()> {
    pyo3::Python::with_gil(|py| {
        let data = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];

        let arr = PyArray::from_array(py, &data);
        let iter = NpySingleIterBuilder::readonly(arr.readonly()).build()?;

        // The order of iteration is not specified, so we should restrict ourselves
        // to tests that don't verify a given order.
        assert_approx_eq!(iter.sum::<f64>(), 15.0);
        Ok(())
    })
}

#[test]
fn mutable_iter() -> PyResult<()> {
    let data = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    pyo3::Python::with_gil(|py| {
        let arr = PyArray::from_array(py, &data);
        let iter = unsafe { NpySingleIterBuilder::readwrite(arr).build()? };
        for elem in iter {
            *elem *= 2.0;
        }
        let iter = NpySingleIterBuilder::readonly(arr.readonly()).build()?;
        assert_approx_eq!(iter.sum::<f64>(), 30.0);
        Ok(())
    })
}

#[test]
fn multiiter_rr() -> PyResult<()> {
    let data1 = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let data2 = array![[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]];

    pyo3::Python::with_gil(|py| {
        let arr1 = PyArray::from_array(py, &data1);
        let arr2 = PyArray::from_array(py, &data2);
        let iter = NpyMultiIterBuilder::new()
            .add_readonly(arr1.readonly())
            .add_readonly(arr2.readonly())
            .build()
            .map_err(|e| e.print(py))
            .unwrap();

        let mut sum = 0.0;
        for (x, y) in iter {
            sum += *x * *y;
        }

        assert_approx_eq!(sum, 145.0);
        Ok(())
    })
}

#[test]
fn multiiter_rw() -> PyResult<()> {
    let data1 = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let data2 = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

    pyo3::Python::with_gil(|py| {
        let arr1 = PyArray::from_array(py, &data1);
        let arr2 = PyArray::from_array(py, &data2);
        let iter = unsafe {
            NpyMultiIterBuilder::new()
                .add_readonly(arr1.readonly())
                .add_readwrite(arr2)
                .build()?
        };

        for (x, y) in iter {
            *y = *x * 2.0;
        }

        let iter = NpyMultiIterBuilder::new()
            .add_readonly(arr1.readonly())
            .add_readonly(arr2.readonly())
            .build()?;

        for (x, y) in iter {
            assert_approx_eq!(*x * 2.0, *y);
        }

        Ok(())
    })
}

#[test]
fn single_iter_size_hint_len() {
    Python::with_gil(|py| {
        let arr = pyarray![py, [0, 1], [2, 3], [4, 5]];

        let mut iter = NpySingleIterBuilder::readonly(arr.readonly())
            .build()
            .unwrap();

        for len in (1..=6).rev() {
            assert_eq!(iter.len(), len);
            assert_eq!(iter.size_hint(), (len, Some(len)));
            assert!(iter.next().is_some());
        }

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    });
}

#[test]
fn multi_iter_size_hint_len() {
    Python::with_gil(|py| {
        let arr1 = pyarray![py, [0, 1], [2, 3], [4, 5]];
        let arr2 = pyarray![py, [0, 0], [0, 0], [0, 0]];

        let mut iter = NpyMultiIterBuilder::new()
            .add_readonly(arr1.readonly())
            .add_readonly(arr2.readonly())
            .build()
            .unwrap();

        for len in (1..=6).rev() {
            assert_eq!(iter.len(), len);
            assert_eq!(iter.size_hint(), (len, Some(len)));
            assert!(iter.next().is_some());
        }

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());
    });
}

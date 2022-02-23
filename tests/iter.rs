#![allow(deprecated)]

use numpy::{pyarray, NpyMultiIterBuilder, NpySingleIterBuilder};
use pyo3::Python;

// The order of iteration is not specified, so we should restrict ourselves
// to tests that don't verify a given order.

#[test]
fn readonly_iter() {
    Python::with_gil(|py| {
        let arr = pyarray![py, [0, 1], [2, 3], [4, 5]];

        let iter = NpySingleIterBuilder::readonly(arr.readonly())
            .build()
            .unwrap();

        assert_eq!(iter.sum::<i32>(), 15);
    });
}

#[test]
fn mutable_iter() {
    Python::with_gil(|py| {
        let arr = pyarray![py, [0, 1], [2, 3], [4, 5]];

        let iter = unsafe { NpySingleIterBuilder::readwrite(arr).build().unwrap() };

        for elem in iter {
            *elem *= 2;
        }

        let iter = NpySingleIterBuilder::readonly(arr.readonly())
            .build()
            .unwrap();

        assert_eq!(iter.sum::<i32>(), 30);
    });
}

#[test]
fn multiiter_rr() {
    Python::with_gil(|py| {
        let arr1 = pyarray![py, [0, 1], [2, 3], [4, 5]];
        let arr2 = pyarray![py, [6, 7], [8, 9], [10, 11]];

        let iter = NpyMultiIterBuilder::new()
            .add_readonly(arr1.readonly())
            .add_readonly(arr2.readonly())
            .build()
            .unwrap();

        assert_eq!(iter.map(|(x, y)| *x * *y).sum::<i32>(), 145);
    });
}

#[test]
fn multiiter_rw() {
    Python::with_gil(|py| {
        let arr1 = pyarray![py, [0, 1], [2, 3], [4, 5]];
        let arr2 = pyarray![py, [0, 0], [0, 0], [0, 0]];

        let iter = unsafe {
            NpyMultiIterBuilder::new()
                .add_readonly(arr1.readonly())
                .add_readwrite(arr2)
                .build()
                .unwrap()
        };

        for (x, y) in iter {
            *y = *x * 2;
        }

        let iter = NpyMultiIterBuilder::new()
            .add_readonly(arr1.readonly())
            .add_readonly(arr2.readonly())
            .build()
            .unwrap();

        for (x, y) in iter {
            assert_eq!(*x * 2, *y);
        }
    });
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

use crate::npyffi::{NPY_CASTING, NPY_ORDER};
use crate::{Element, PyArray, PY_ARRAY_API};
use ndarray::{Dimension, IxDyn};
use pyo3::{AsPyPointer, FromPyPointer, PyAny, PyNativeType, PyResult};
use std::ffi::CStr;

/// Return the inner product of two arrays.
///
/// # Example
/// ```
/// pyo3::Python::with_gil(|py| {
///     let array = numpy::pyarray![py, 1, 2, 3];
///     let inner: &numpy::PyArray0::<_> = numpy::inner(array, array).unwrap();
///     assert_eq!(inner.item(), 14);
/// });
/// ```
pub fn inner<'py, T, DIN1, DIN2, DOUT>(
    array1: &'py PyArray<T, DIN1>,
    array2: &'py PyArray<T, DIN2>,
) -> PyResult<&'py PyArray<T, DOUT>>
where
    DIN1: Dimension,
    DIN2: Dimension,
    DOUT: Dimension,
    T: Element,
{
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_InnerProduct(array1.as_ptr(), array2.as_ptr());
        PyAny::from_owned_ptr_or_err(array1.py(), result)?
    };
    obj.extract()
}

/// Return the dot product of two arrays.
///
/// # Example
/// ```
/// pyo3::Python::with_gil(|py| {
///     let a = numpy::pyarray![py, [1, 0], [0, 1]];
///     let b = numpy::pyarray![py, [4, 1], [2, 2]];
///     let dot: &numpy::PyArray2::<_> = numpy::dot(a, b).unwrap();
///     assert_eq!(
///         dot.readonly().as_array(),
///         ndarray::array![[4, 1], [2, 2]]
///     );
/// });
/// ```
pub fn dot<'py, T, DIN1, DIN2, DOUT>(
    array1: &'py PyArray<T, DIN1>,
    array2: &'py PyArray<T, DIN2>,
) -> PyResult<&'py PyArray<T, DOUT>>
where
    DIN1: Dimension,
    DIN2: Dimension,
    DOUT: Dimension,
    T: Element,
{
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_MatrixProduct(array1.as_ptr(), array2.as_ptr());
        PyAny::from_owned_ptr_or_err(array1.py(), result)?
    };
    obj.extract()
}

/// Return the Einstein summation convention of given tensors.
///
/// We also provide the [einsum macro](./macro.einsum.html).
pub fn einsum_impl<'py, T, DOUT>(
    subscripts: &str,
    arrays: &[&'py PyArray<T, IxDyn>],
) -> PyResult<&'py PyArray<T, DOUT>>
where
    DOUT: Dimension,
    T: Element,
{
    let subscripts: std::borrow::Cow<CStr> = if subscripts.ends_with("\0") {
        CStr::from_bytes_with_nul(subscripts.as_bytes())
            .unwrap()
            .into()
    } else {
        std::ffi::CString::new(subscripts).unwrap().into()
    };
    let obj = unsafe {
        let result = PY_ARRAY_API.PyArray_EinsteinSum(
            subscripts.as_ptr() as _,
            arrays.len() as _,
            arrays.as_ptr() as _,
            std::ptr::null_mut(),
            NPY_ORDER::NPY_KEEPORDER,
            NPY_CASTING::NPY_NO_CASTING,
            std::ptr::null_mut(),
        );
        PyAny::from_owned_ptr_or_err(arrays[0].py(), result)?
    };
    obj.extract()
}

/// Return the Einstein summation convention of given tensors.
///
/// For more about the Einstein summation convention, you may reffer to
/// [the numpy document](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
///
/// # Example
/// ```
/// pyo3::Python::with_gil(|py| {
///     let a = numpy::PyArray::arange(py, 0, 2 * 3 * 4, 1).reshape([2, 3, 4]).unwrap();
///     let b = numpy::pyarray![py, [20, 30], [40, 50], [60, 70]];
///     let einsum = numpy::einsum!("ijk,ji->ik", a, b).unwrap();
///     assert_eq!(
///         einsum.readonly().as_array(),
///         ndarray::array![[640,  760,  880, 1000], [2560, 2710, 2860, 3010]]
///     );
/// });
/// ```
#[macro_export]
macro_rules! einsum {
    ($subscripts: literal $(,$array: ident)+ $(,)*) => {{
        let arrays = [$($array.to_dyn(),)+];
        unsafe { $crate::einsum_impl(concat!($subscripts, "\0"), &arrays) }
    }};
}

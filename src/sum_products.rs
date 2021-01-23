use crate::npyffi::{PyArrayObject, NPY_CASTING, NPY_ORDER};
use crate::{Element, PyArray, PY_ARRAY_API};
use ndarray::Dimension;
use pyo3::{AsPyPointer, FromPyPointer, PyAny, PyNativeType, PyResult};

/// Return the inner product of two arrays.
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
    let result = unsafe { PY_ARRAY_API.PyArray_InnerProduct(array1.as_ptr(), array2.as_ptr()) };
    let obj = unsafe { PyAny::from_owned_ptr_or_err(array1.py(), result)? };
    obj.extract()
}

/// Return the dot product of two arrays.
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
    let result = unsafe { PY_ARRAY_API.PyArray_MatrixProduct(array1.as_ptr(), array2.as_ptr()) };
    let obj = unsafe { PyAny::from_owned_ptr_or_err(array1.py(), result)? };
    obj.extract()
}

pub unsafe fn einsum_impl<'py, T, DIN, DOUT>(
    dummy_array: &'py PyArray<T, DIN>,
    subscripts: &str,
    arrays: &[*mut PyArrayObject],
) -> PyResult<&'py PyArray<T, DOUT>>
where
    DIN: Dimension,
    DOUT: Dimension,
    T: Element,
{
    let subscripts = std::ffi::CStr::from_bytes_with_nul(subscripts.as_bytes()).unwrap();
    let result = PY_ARRAY_API.PyArray_EinsteinSum(
        subscripts.as_ptr() as _,
        arrays.len() as _,
        arrays.as_ptr() as _,
        std::ptr::null_mut(),
        NPY_ORDER::NPY_KEEPORDER,
        NPY_CASTING::NPY_NO_CASTING,
        std::ptr::null_mut(),
    );
    let obj = PyAny::from_owned_ptr_or_err(dummy_array.py(), result)?;
    obj.extract()
}

#[macro_export]
macro_rules! einsum {
    ($subscripts: literal, $first_array: ident $(,$array: ident)* $(,)*) => {{
        let arrays = [$first_array.as_array_ptr(), $($array.as_array_ptr(),)*];
        unsafe { $crate::einsum_impl($first_array, concat!($subscripts, "\0"), &arrays) }
    }};
}

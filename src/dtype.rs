use std::mem::size_of;
use std::os::raw::{c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong, c_ushort};

use cfg_if::cfg_if;
use num_traits::{Bounded, Zero};
use pyo3::{ffi, prelude::*, pyobject_native_type_core, types::PyType, AsPyPointer, PyNativeType};

use crate::npyffi::{NpyTypes, PyArray_Descr, NPY_TYPES, PY_ARRAY_API};

pub use num_complex::{Complex32, Complex64};

/// Binding of [`numpy.dtype`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html).
///
/// # Example
/// ```
/// use pyo3::types::IntoPyDict;
/// pyo3::Python::with_gil(|py| {
///     let locals = [("np", numpy::get_array_module(py).unwrap())].into_py_dict(py);
///     let dtype: &numpy::PyArrayDescr = py
///         .eval("np.array([1, 2, 3.0]).dtype", Some(locals), None)
///         .unwrap()
///         .downcast()
///         .unwrap();
///     assert!(dtype.is_equiv_to(numpy::PyArrayDescr::of::<f64>(py)));
/// });
/// ```
pub struct PyArrayDescr(PyAny);

pyobject_native_type_core!(
    PyArrayDescr,
    *PY_ARRAY_API.get_type_object(NpyTypes::PyArrayDescr_Type),
    #module=Some("numpy"),
    #checkfunction=arraydescr_check
);

unsafe fn arraydescr_check(op: *mut ffi::PyObject) -> c_int {
    ffi::PyObject_TypeCheck(
        op,
        PY_ARRAY_API.get_type_object(NpyTypes::PyArrayDescr_Type),
    )
}

impl PyArrayDescr {
    /// Returns `self` as `*mut PyArray_Descr`.
    pub fn as_dtype_ptr(&self) -> *mut PyArray_Descr {
        self.as_ptr() as _
    }

    /// Returns `self` as `*mut PyArray_Descr` while increasing the reference count.
    ///
    /// Useful in cases where the descriptor is stolen by the API.
    pub fn into_dtype_ptr(&self) -> *mut PyArray_Descr {
        self.into_ptr() as _
    }

    /// Returns the internal `PyType` that this `dtype` holds.
    ///
    /// # Example
    /// ```
    /// pyo3::Python::with_gil(|py| {
    ///    let array = numpy::PyArray::from_vec(py, vec![0.0, 1.0, 2.0f64]);
    ///    let dtype = array.dtype();
    ///    assert_eq!(dtype.get_type().name().unwrap().to_string(), "float64");
    /// });
    /// ```
    pub fn get_type(&self) -> &PyType {
        let dtype_type_ptr = unsafe { *self.as_dtype_ptr() }.typeobj;
        unsafe { PyType::from_type_ptr(self.py(), dtype_type_ptr) }
    }

    /// Shortcut for creating a descriptor of 'object' type.
    pub fn object(py: Python) -> &Self {
        Self::from_npy_type(py, NPY_TYPES::NPY_OBJECT)
    }

    /// Returns the type descriptor for a registered type.
    pub fn of<T: Element>(py: Python) -> &Self {
        T::get_dtype(py)
    }

    /// Returns true if two type descriptors are equivalent.
    pub fn is_equiv_to(&self, other: &Self) -> bool {
        unsafe { PY_ARRAY_API.PyArray_EquivTypes(self.as_dtype_ptr(), other.as_dtype_ptr()) != 0 }
    }

    fn from_npy_type(py: Python, npy_type: NPY_TYPES) -> &Self {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(npy_type as _);
            py.from_owned_ptr(descr as _)
        }
    }

    /// Retrieves the
    /// [enumerated type](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types)
    /// for this type descriptor.
    pub fn get_typenum(&self) -> c_int {
        unsafe { *self.as_dtype_ptr() }.type_num
    }
}

/// Represents that a type can be an element of `PyArray`.
///
/// Currently, only integer/float/complex types are supported.
/// If you come up with a nice implementation for some other types, we're happy to receive your PR :)
/// You may refer to the [numpy document](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types)
/// for all types that numpy supports.
///
/// # Safety
///
/// A type `T` that implements this trait should be safe when managed in numpy array,
/// thus implementing this trait is marked unsafe.
/// This means that all data types except for `DataType::Object` are assumed to be trivially copyable.
/// Furthermore, it is assumed that for `DataType::Object` the elements are pointers into the Python heap
/// and that the corresponding `Clone` implemenation will never panic as it only increases the reference count.
///
/// # Custom element types
///
/// You can implement this trait to manage arrays of custom element types, but they still need to be stored
/// on Python's heap using PyO3's [Py](pyo3::Py) type.
///
/// ```
/// use numpy::{ndarray::Array2, Element, PyArray, PyArrayDescr, ToPyArray};
/// use pyo3::{pyclass, Py, Python};
///
/// #[pyclass]
/// pub struct CustomElement;
///
/// // The transparent wrapper is necessary as one cannot implement
/// // a foreign trait (`Element`) on a foreign type (`Py`) directly.
/// #[derive(Clone)]
/// #[repr(transparent)]
/// pub struct Wrapper(pub Py<CustomElement>);
///
/// unsafe impl Element for Wrapper {
///     const IS_COPY: bool = false;
///
///     fn get_dtype(py: Python) -> &PyArrayDescr {
///         PyArrayDescr::object(py)
///     }
/// }
///
/// Python::with_gil(|py| {
///     let array = Array2::<Wrapper>::from_shape_fn((2, 3), |(_i, _j)| {
///         Wrapper(Py::new(py, CustomElement).unwrap())
///     });
///
///     let _array: &PyArray<Wrapper, _> = array.to_pyarray(py);
/// });
/// ```
pub unsafe trait Element: Clone + Send {
    /// Flag that indicates whether this type is trivially copyable.
    ///
    /// It should be set to true for all trivially copyable types (like scalar types
    /// and record/array types only containing trivially copyable fields and elements).
    ///
    /// This flag should *always* be set to `false` for object types or record types
    /// that contain object-type fields.
    const IS_COPY: bool;

    /// Returns the associated array descriptor ("dtype") for the given type.
    fn get_dtype(py: Python) -> &PyArrayDescr;
}

fn npy_int_type_lookup<T, T0, T1, T2>(npy_types: [NPY_TYPES; 3]) -> NPY_TYPES {
    // `npy_common.h` defines the integer aliases. In order, it checks:
    // NPY_BITSOF_LONG, NPY_BITSOF_LONGLONG, NPY_BITSOF_INT, NPY_BITSOF_SHORT, NPY_BITSOF_CHAR
    // and assigns the alias to the first matching size, so we should check in this order.
    match size_of::<T>() {
        x if x == size_of::<T0>() => npy_types[0],
        x if x == size_of::<T1>() => npy_types[1],
        x if x == size_of::<T2>() => npy_types[2],
        _ => panic!("Unable to match integer type descriptor: {:?}", npy_types),
    }
}

fn npy_int_type<T: Bounded + Zero + Sized + PartialEq>() -> NPY_TYPES {
    let is_unsigned = T::min_value() == T::zero();
    let bit_width = size_of::<T>() << 3;

    match (is_unsigned, bit_width) {
        (false, 8) => NPY_TYPES::NPY_BYTE,
        (false, 16) => NPY_TYPES::NPY_SHORT,
        (false, 32) => npy_int_type_lookup::<i32, c_long, c_int, c_short>([
            NPY_TYPES::NPY_LONG,
            NPY_TYPES::NPY_INT,
            NPY_TYPES::NPY_SHORT,
        ]),
        (false, 64) => npy_int_type_lookup::<i64, c_long, c_longlong, c_int>([
            NPY_TYPES::NPY_LONG,
            NPY_TYPES::NPY_LONGLONG,
            NPY_TYPES::NPY_INT,
        ]),
        (true, 8) => NPY_TYPES::NPY_UBYTE,
        (true, 16) => NPY_TYPES::NPY_USHORT,
        (true, 32) => npy_int_type_lookup::<u32, c_ulong, c_uint, c_ushort>([
            NPY_TYPES::NPY_ULONG,
            NPY_TYPES::NPY_UINT,
            NPY_TYPES::NPY_USHORT,
        ]),
        (true, 64) => npy_int_type_lookup::<u64, c_ulong, c_ulonglong, c_uint>([
            NPY_TYPES::NPY_ULONG,
            NPY_TYPES::NPY_ULONGLONG,
            NPY_TYPES::NPY_UINT,
        ]),
        _ => unreachable!(),
    }
}

macro_rules! impl_element_scalar {
    (@impl: $ty:ty, $npy_type:expr $(,#[$meta:meta])*) => {
        $(#[$meta])*
        unsafe impl Element for $ty {
            const IS_COPY: bool = true;
            fn get_dtype(py: Python) -> &PyArrayDescr {
                PyArrayDescr::from_npy_type(py, $npy_type)
            }
        }
    };
    ($ty:ty, $npy_type:ident $(,#[$meta:meta])*) => {
        impl_element_scalar!(@impl: $ty, NPY_TYPES::$npy_type $(,#[$meta])*);
    };
    ($ty:ty $(,#[$meta:meta])*) => {
        impl_element_scalar!(@impl: $ty, npy_int_type::<$ty>() $(,#[$meta])*);
    };
}

impl_element_scalar!(bool, NPY_BOOL);
impl_element_scalar!(i8);
impl_element_scalar!(i16);
impl_element_scalar!(i32);
impl_element_scalar!(i64);
impl_element_scalar!(u8);
impl_element_scalar!(u16);
impl_element_scalar!(u32);
impl_element_scalar!(u64);
impl_element_scalar!(f32, NPY_FLOAT);
impl_element_scalar!(f64, NPY_DOUBLE);
impl_element_scalar!(Complex32, NPY_CFLOAT,
    #[doc = "Complex type with `f32` components which maps to `np.csingle` (`np.complex64`)."]);
impl_element_scalar!(Complex64, NPY_CDOUBLE,
    #[doc = "Complex type with `f64` components which maps to `np.cdouble` (`np.complex128`)."]);

cfg_if! {
    if #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))] {
        impl_element_scalar!(usize);
        impl_element_scalar!(isize);
    }
}

unsafe impl Element for PyObject {
    const IS_COPY: bool = false;

    fn get_dtype(py: Python) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::{Complex32, Complex64, Element, PyArrayDescr};

    #[test]
    fn test_dtype_names() {
        fn type_name<T: Element>(py: pyo3::Python) -> &str {
            PyArrayDescr::of::<T>(py).get_type().name().unwrap()
        }
        pyo3::Python::with_gil(|py| {
            assert_eq!(type_name::<bool>(py), "bool_");
            assert_eq!(type_name::<i8>(py), "int8");
            assert_eq!(type_name::<i16>(py), "int16");
            assert_eq!(type_name::<i32>(py), "int32");
            assert_eq!(type_name::<i64>(py), "int64");
            assert_eq!(type_name::<u8>(py), "uint8");
            assert_eq!(type_name::<u16>(py), "uint16");
            assert_eq!(type_name::<u32>(py), "uint32");
            assert_eq!(type_name::<u64>(py), "uint64");
            assert_eq!(type_name::<f32>(py), "float32");
            assert_eq!(type_name::<f64>(py), "float64");
            assert_eq!(type_name::<Complex32>(py), "complex64");
            assert_eq!(type_name::<Complex64>(py), "complex128");
            match size_of::<usize>() {
                32 => {
                    assert_eq!(type_name::<usize>(py), "uint32");
                    assert_eq!(type_name::<isize>(py), "int32");
                }
                64 => {
                    assert_eq!(type_name::<usize>(py), "uint64");
                    assert_eq!(type_name::<isize>(py), "int64");
                }
                _ => {}
            }
        });
    }
}

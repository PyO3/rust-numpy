use std::mem::size_of;
use std::os::raw::{c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong, c_ushort};

use cfg_if::cfg_if;
use num_traits::{Bounded, Zero};
use pyo3::{ffi, prelude::*, pyobject_native_type_core, types::PyType, AsPyPointer, PyNativeType};

use crate::npyffi::{NpyTypes, PyArray_Descr, NPY_TYPES, PY_ARRAY_API};

pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;

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
///     assert_eq!(dtype.get_datatype().unwrap(), numpy::DataType::Float64);
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

    /// Returns the data type as `DataType` enum.
    pub fn get_datatype(&self) -> Option<DataType> {
        DataType::from_typenum(self.get_typenum())
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

/// Represents NumPy data type.
///
/// This is an incomplete counterpart of
/// [Enumerated Types](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types)
/// in numpy C-API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    Complex32,
    Complex64,
    Object,
}

impl DataType {
    /// Convert `self` into an
    /// [enumerated type](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    pub fn into_typenum(self) -> c_int {
        self.into_npy_type() as _
    }

    /// Construct the data type from an
    /// [enumerated type](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    pub fn from_typenum(typenum: c_int) -> Option<Self> {
        Some(match typenum {
            x if x == NPY_TYPES::NPY_BOOL as c_int => DataType::Bool,
            x if x == NPY_TYPES::NPY_BYTE as c_int => DataType::Int8,
            x if x == NPY_TYPES::NPY_SHORT as c_int => DataType::Int16,
            x if x == NPY_TYPES::NPY_INT as c_int => Self::integer::<c_int>()?,
            x if x == NPY_TYPES::NPY_LONG as c_int => Self::integer::<c_long>()?,
            x if x == NPY_TYPES::NPY_LONGLONG as c_int => Self::integer::<c_longlong>()?,
            x if x == NPY_TYPES::NPY_UBYTE as c_int => DataType::Uint8,
            x if x == NPY_TYPES::NPY_USHORT as c_int => DataType::Uint16,
            x if x == NPY_TYPES::NPY_UINT as c_int => Self::integer::<c_uint>()?,
            x if x == NPY_TYPES::NPY_ULONG as c_int => Self::integer::<c_ulong>()?,
            x if x == NPY_TYPES::NPY_ULONGLONG as c_int => Self::integer::<c_ulonglong>()?,
            x if x == NPY_TYPES::NPY_FLOAT as c_int => DataType::Float32,
            x if x == NPY_TYPES::NPY_DOUBLE as c_int => DataType::Float64,
            x if x == NPY_TYPES::NPY_CFLOAT as c_int => DataType::Complex32,
            x if x == NPY_TYPES::NPY_CDOUBLE as c_int => DataType::Complex64,
            x if x == NPY_TYPES::NPY_OBJECT as c_int => DataType::Object,
            _ => return None,
        })
    }

    #[inline]
    fn integer<T: Bounded + Zero + Sized + PartialEq>() -> Option<Self> {
        let is_unsigned = T::min_value() == T::zero();
        let bit_width = size_of::<T>() << 3;
        Some(match (is_unsigned, bit_width) {
            (false, 8) => Self::Int8,
            (false, 16) => Self::Int16,
            (false, 32) => Self::Int32,
            (false, 64) => Self::Int64,
            (true, 8) => Self::Uint8,
            (true, 16) => Self::Uint16,
            (true, 32) => Self::Uint32,
            (true, 64) => Self::Uint64,
            _ => return None,
        })
    }

    fn into_npy_type(self) -> NPY_TYPES {
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

        match self {
            DataType::Bool => NPY_TYPES::NPY_BOOL,
            DataType::Int8 => NPY_TYPES::NPY_BYTE,
            DataType::Int16 => NPY_TYPES::NPY_SHORT,
            DataType::Int32 => npy_int_type_lookup::<i32, c_long, c_int, c_short>([
                NPY_TYPES::NPY_LONG,
                NPY_TYPES::NPY_INT,
                NPY_TYPES::NPY_SHORT,
            ]),
            DataType::Int64 => npy_int_type_lookup::<i64, c_long, c_longlong, c_int>([
                NPY_TYPES::NPY_LONG,
                NPY_TYPES::NPY_LONGLONG,
                NPY_TYPES::NPY_INT,
            ]),
            DataType::Uint8 => NPY_TYPES::NPY_UBYTE,
            DataType::Uint16 => NPY_TYPES::NPY_USHORT,
            DataType::Uint32 => npy_int_type_lookup::<u32, c_ulong, c_uint, c_ushort>([
                NPY_TYPES::NPY_ULONG,
                NPY_TYPES::NPY_UINT,
                NPY_TYPES::NPY_USHORT,
            ]),
            DataType::Uint64 => npy_int_type_lookup::<u64, c_ulong, c_ulonglong, c_uint>([
                NPY_TYPES::NPY_ULONG,
                NPY_TYPES::NPY_ULONGLONG,
                NPY_TYPES::NPY_UINT,
            ]),
            DataType::Float32 => NPY_TYPES::NPY_FLOAT,
            DataType::Float64 => NPY_TYPES::NPY_DOUBLE,
            DataType::Complex32 => NPY_TYPES::NPY_CFLOAT,
            DataType::Complex64 => NPY_TYPES::NPY_CDOUBLE,
            DataType::Object => NPY_TYPES::NPY_OBJECT,
        }
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
/// use numpy::{ndarray::Array2, DataType, Element, PyArray, PyArrayDescr, ToPyArray};
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
///     const DATA_TYPE: DataType = DataType::Object;
///
///     fn is_same_type(dtype: &PyArrayDescr) -> bool {
///         dtype.get_datatype() == Some(DataType::Object)
///     }
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
    /// `DataType` corresponding to this type.
    const DATA_TYPE: DataType;

    /// Returns if the give `dtype` is convertible to `Self` in Rust.
    fn is_same_type(dtype: &PyArrayDescr) -> bool;

    /// Create `dtype`.
    fn get_dtype(py: Python) -> &PyArrayDescr;
}

macro_rules! impl_num_element {
    ($ty:ty, $data_type:expr) => {
        unsafe impl Element for $ty {
            const DATA_TYPE: DataType = $data_type;

            fn is_same_type(dtype: &PyArrayDescr) -> bool {
                dtype.get_datatype() == Some($data_type)
            }

            fn get_dtype(py: Python) -> &PyArrayDescr {
                PyArrayDescr::from_npy_type(py, $data_type.into_npy_type())
            }
        }
    };
}

impl_num_element!(bool, DataType::Bool);
impl_num_element!(i8, DataType::Int8);
impl_num_element!(i16, DataType::Int16);
impl_num_element!(i32, DataType::Int32);
impl_num_element!(i64, DataType::Int64);
impl_num_element!(u8, DataType::Uint8);
impl_num_element!(u16, DataType::Uint16);
impl_num_element!(u32, DataType::Uint32);
impl_num_element!(u64, DataType::Uint64);
impl_num_element!(f32, DataType::Float32);
impl_num_element!(f64, DataType::Float64);
impl_num_element!(c32, DataType::Complex32);
impl_num_element!(c64, DataType::Complex64);

cfg_if! {
    if #[cfg(target_pointer_width = "64")] {
        impl_num_element!(usize, DataType::Uint64);
    } else if #[cfg(target_pointer_width = "32")] {
        impl_num_element!(usize, DataType::Uint32);
    }
}

unsafe impl Element for PyObject {
    const DATA_TYPE: DataType = DataType::Object;

    fn is_same_type(dtype: &PyArrayDescr) -> bool {
        dtype.get_typenum() == NPY_TYPES::NPY_OBJECT as i32
    }

    fn get_dtype(py: Python) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

#[cfg(test)]
mod tests {
    use cfg_if::cfg_if;

    use super::{c32, c64, Element, PyArrayDescr};

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
            assert_eq!(type_name::<c32>(py), "complex64");
            assert_eq!(type_name::<c64>(py), "complex128");
            cfg_if! {
                if #[cfg(target_pointer_width = "64")] {
                    assert_eq!(type_name::<usize>(py), "uint64");
                } else if #[cfg(target_pointer_width = "32")] {
                    assert_eq!(type_name::<usize>(py), "uint32");
                }
            }
        })
    }
}

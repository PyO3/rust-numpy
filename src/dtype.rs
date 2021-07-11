use crate::npyffi::{NpyTypes, PyArray_Descr, NPY_TYPES, PY_ARRAY_API};
pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{AsPyPointer, PyNativeType};
use std::os::raw::c_int;

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

    fn from_npy_type(py: Python, npy_type: NPY_TYPES) -> &Self {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(npy_type as i32);
            py.from_owned_ptr(descr as _)
        }
    }

    fn get_typenum(&self) -> std::os::raw::c_int {
        unsafe { *self.as_dtype_ptr() }.type_num
    }
}

/// Represents numpy data type.
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
    Unicode,
}

impl DataType {
    /// Construct `DataType` from
    /// [Enumerated Types](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    pub fn from_typenum(typenum: c_int) -> Option<Self> {
        Some(match typenum {
            x if x == NPY_TYPES::NPY_BOOL as i32 => DataType::Bool,
            x if x == NPY_TYPES::NPY_BYTE as i32 => DataType::Int8,
            x if x == NPY_TYPES::NPY_SHORT as i32 => DataType::Int16,
            x if x == NPY_TYPES::NPY_INT as i32 => DataType::Int32,
            x if x == NPY_TYPES::NPY_LONG as i32 => return DataType::from_clong(false),
            x if x == NPY_TYPES::NPY_LONGLONG as i32 => DataType::Int64,
            x if x == NPY_TYPES::NPY_UBYTE as i32 => DataType::Uint8,
            x if x == NPY_TYPES::NPY_USHORT as i32 => DataType::Uint16,
            x if x == NPY_TYPES::NPY_UINT as i32 => DataType::Uint32,
            x if x == NPY_TYPES::NPY_ULONG as i32 => return DataType::from_clong(true),
            x if x == NPY_TYPES::NPY_ULONGLONG as i32 => DataType::Uint64,
            x if x == NPY_TYPES::NPY_FLOAT as i32 => DataType::Float32,
            x if x == NPY_TYPES::NPY_DOUBLE as i32 => DataType::Float64,
            x if x == NPY_TYPES::NPY_CFLOAT as i32 => DataType::Complex32,
            x if x == NPY_TYPES::NPY_CDOUBLE as i32 => DataType::Complex64,
            x if x == NPY_TYPES::NPY_OBJECT as i32 => DataType::Object,
            x if x == NPY_TYPES::NPY_UNICODE as i32 => DataType::Unicode,
            _ => return None,
        })
    }

    /// Convert `self` into
    /// [Enumerated Types](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    pub fn into_ctype(self) -> NPY_TYPES {
        match self {
            DataType::Bool => NPY_TYPES::NPY_BOOL,
            DataType::Int8 => NPY_TYPES::NPY_BYTE,
            DataType::Int16 => NPY_TYPES::NPY_SHORT,
            DataType::Int32 => NPY_TYPES::NPY_INT,
            #[cfg(all(target_pointer_width = "64", not(windows)))]
            DataType::Int64 => NPY_TYPES::NPY_LONG,
            #[cfg(any(target_pointer_width = "32", windows))]
            DataType::Int64 => NPY_TYPES::NPY_LONGLONG,
            DataType::Uint8 => NPY_TYPES::NPY_UBYTE,
            DataType::Uint16 => NPY_TYPES::NPY_USHORT,
            DataType::Uint32 => NPY_TYPES::NPY_UINT,
            DataType::Uint64 => NPY_TYPES::NPY_ULONGLONG,
            DataType::Float32 => NPY_TYPES::NPY_FLOAT,
            DataType::Float64 => NPY_TYPES::NPY_DOUBLE,
            DataType::Complex32 => NPY_TYPES::NPY_CFLOAT,
            DataType::Complex64 => NPY_TYPES::NPY_CDOUBLE,
            DataType::Object => NPY_TYPES::NPY_OBJECT,
            DataType::Unicode => NPY_TYPES::NPY_UNICODE,
        }
    }

    #[inline(always)]
    fn from_clong(is_usize: bool) -> Option<Self> {
        if cfg!(any(target_pointer_width = "32", windows)) {
            Some(if is_usize {
                DataType::Uint32
            } else {
                DataType::Int32
            })
        } else if cfg!(all(target_pointer_width = "64", not(windows))) {
            Some(if is_usize {
                DataType::Uint64
            } else {
                DataType::Int64
            })
        } else {
            None
        }
    }
}

/// Represents that a type can be an element of `PyArray`.
pub trait Element: Clone + Send {
    /// `DataType` corresponding to this type.
    const DATA_TYPE: DataType;

    /// Returns if the give `dtype` is convertible to `Self` in Rust.
    fn is_same_type(dtype: &PyArrayDescr) -> bool;

    /// Returns the corresponding
    /// [Enumerated Type](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    #[inline]
    fn npy_type() -> NPY_TYPES {
        Self::DATA_TYPE.into_ctype()
    }

    /// Create `dtype`.
    fn get_dtype(py: Python) -> &PyArrayDescr {
        PyArrayDescr::from_npy_type(py, Self::npy_type())
    }

    /// itemsize in bytes
    fn item_size() -> usize {
        0
    }
}

macro_rules! impl_num_element {
    ($t:ty, $npy_dat_t:ident $(,$npy_types: ident)+) => {
        impl Element for $t {
            const DATA_TYPE: DataType = DataType::$npy_dat_t;
            fn is_same_type(dtype: &PyArrayDescr) -> bool {
                $(dtype.get_typenum() == NPY_TYPES::$npy_types as i32 ||)+ false
            }
        }
    };
    ($t:ty, $npy_dat_t:ident, $npy_itemsize:expr $(,$npy_types: ident)+) => {
        impl Element for $t {
            const DATA_TYPE: DataType = DataType::$npy_dat_t;
            fn is_same_type(dtype: &PyArrayDescr) -> bool {
                $(dtype.get_typenum() == NPY_TYPES::$npy_types as i32 ||)+ false
            }

            fn item_size() -> usize {
                $npy_itemsize
            }
        }
    };
}

impl_num_element!(bool, Bool, NPY_BOOL);
impl_num_element!(i8, Int8, NPY_BYTE);
impl_num_element!(i16, Int16, NPY_SHORT);
impl_num_element!(u8, Uint8, NPY_UBYTE);
impl_num_element!(u16, Uint16, NPY_USHORT);
impl_num_element!(f32, Float32, NPY_FLOAT);
impl_num_element!(f64, Float64, NPY_DOUBLE);
impl_num_element!(c32, Complex32, NPY_CFLOAT);
impl_num_element!(c64, Complex64, NPY_CDOUBLE);
impl_num_element!(String, Unicode, std::mem::size_of::<String>(), NPY_UNICODE);
impl_num_element!(&str, Unicode, std::mem::size_of::<&str>(), NPY_UNICODE);

cfg_if! {
    if #[cfg(all(target_pointer_width = "64", windows))] {
            impl_num_element!(usize, Uint64, NPY_ULONGLONG);
    } else if #[cfg(all(target_pointer_width = "64", not(windows)))] {
            impl_num_element!(usize, Uint64, NPY_ULONG, NPY_ULONGLONG);
    } else if #[cfg(all(target_pointer_width = "32", windows))] {
            impl_num_element!(usize, Uint32, NPY_UINT, NPY_ULONG);
    } else if #[cfg(all(target_pointer_width = "32", not(windows)))] {
            impl_num_element!(usize, Uint32, NPY_UINT);
    }
}
cfg_if! {
    if #[cfg(any(target_pointer_width = "32", windows))] {
        impl_num_element!(i32, Int32, NPY_INT, NPY_LONG);
        impl_num_element!(u32, Uint32, NPY_UINT, NPY_ULONG);
        impl_num_element!(i64, Int64, NPY_LONGLONG);
        impl_num_element!(u64, Uint64, NPY_ULONGLONG);
    } else if #[cfg(all(target_pointer_width = "64", not(windows)))] {
        impl_num_element!(i32, Int32, NPY_INT);
        impl_num_element!(u32, Uint32, NPY_UINT);
        impl_num_element!(i64, Int64, NPY_LONG, NPY_LONGLONG);
        impl_num_element!(u64, Uint64, NPY_ULONG, NPY_ULONGLONG);
    }
}

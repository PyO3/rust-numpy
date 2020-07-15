//! Implements conversion utitlities.
/// alias of Complex32
pub use num_complex::Complex32 as c32;
/// alias of Complex64
pub use num_complex::Complex64 as c64;

use super::npyffi::NPY_TYPES;

/// An enum type represents numpy data type.
///
/// This type is mainly for displaying error, and user don't have to use it directly.
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
    pub(crate) fn from_i32(npy_t: i32) -> Option<Self> {
        Some(match npy_t {
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
            _ => return None,
        })
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
    pub fn into_ffi_dtype(self) -> NPY_TYPES {
        match self {
            DataType::Bool => NPY_TYPES::NPY_BOOL,
            DataType::Int8 => NPY_TYPES::NPY_BYTE,
            DataType::Int16 => NPY_TYPES::NPY_SHORT,
            DataType::Int32 => NPY_TYPES::NPY_INT,
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
        }
    }
}

/// Represents that a type can be an element of `PyArray`.
pub trait Element: Clone {
    const DATA_TYPE: DataType;
    fn is_same_type(other: i32) -> bool;
    fn ffi_dtype() -> NPY_TYPES {
        Self::DATA_TYPE.into_ffi_dtype()
    }
}

macro_rules! impl_num_element {
    ($t:ty, $npy_dat_t:ident $(,$npy_types: ident)+) => {
        impl Element for $t {
            const DATA_TYPE: DataType = DataType::$npy_dat_t;
            fn is_same_type(other: i32) -> bool {
                $(other == NPY_TYPES::$npy_types as i32 ||)+ false
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

impl Element for pyo3::PyObject {
    const DATA_TYPE: DataType = DataType::Object;
    fn is_same_type(other: i32) -> bool {
        other == Self::DATA_TYPE as i32
    }
}

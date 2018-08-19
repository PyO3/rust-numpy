pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;

pub use super::npyffi::npy_intp;
pub use super::npyffi::NPY_ORDER;
pub use super::npyffi::NPY_ORDER::{NPY_CORDER, NPY_FORTRANORDER};

use super::npyffi::NPY_TYPES;

/// An enum type represents numpy data type.
///
/// This type is mainly for displaying error, and user don't have to use it directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NpyDataType {
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
    Unsupported,
}

impl NpyDataType {
    pub(crate) fn from_i32(npy_t: i32) -> Self {
        match npy_t {
            x if x == NPY_TYPES::NPY_BOOL as i32 => NpyDataType::Bool,
            x if x == NPY_TYPES::NPY_BYTE as i32 => NpyDataType::Int8,
            x if x == NPY_TYPES::NPY_SHORT as i32 => NpyDataType::Int16,
            x if x == NPY_TYPES::NPY_INT as i32 => NpyDataType::Int32,
            x if x == NPY_TYPES::NPY_LONG as i32 => NpyDataType::from_clong(false),
            x if x == NPY_TYPES::NPY_LONGLONG as i32 => NpyDataType::Int64,
            x if x == NPY_TYPES::NPY_UBYTE as i32 => NpyDataType::Uint8,
            x if x == NPY_TYPES::NPY_USHORT as i32 => NpyDataType::Uint16,
            x if x == NPY_TYPES::NPY_UINT as i32 => NpyDataType::Uint32,
            x if x == NPY_TYPES::NPY_ULONG as i32 => NpyDataType::from_clong(true),
            x if x == NPY_TYPES::NPY_ULONGLONG as i32 => NpyDataType::Uint64,
            x if x == NPY_TYPES::NPY_FLOAT as i32 => NpyDataType::Float32,
            x if x == NPY_TYPES::NPY_DOUBLE as i32 => NpyDataType::Float64,
            x if x == NPY_TYPES::NPY_CFLOAT as i32 => NpyDataType::Complex32,
            x if x == NPY_TYPES::NPY_CDOUBLE as i32 => NpyDataType::Complex64,
            _ => NpyDataType::Unsupported,
        }
    }
    #[inline(always)]
    fn from_clong(is_usize: bool) -> NpyDataType {
        if cfg!(any(target_pointer_width = "32", windows)) {
            if is_usize {
                NpyDataType::Uint32
            } else {
                NpyDataType::Int32
            }
        } else if cfg!(all(target_pointer_width = "64", not(windows))) {
            if is_usize {
                NpyDataType::Uint64
            } else {
                NpyDataType::Int64
            }
        } else {
            NpyDataType::Unsupported
        }
    }
}

pub trait TypeNum: Clone {
    fn is_same_type(other: i32) -> bool;
    fn npy_data_type() -> NpyDataType;
    fn typenum_default() -> i32;
}

macro_rules! impl_type_num {
    ($t:ty, $npy_dat_t:ident $(,$npy_types: ident)+) => {
        impl TypeNum for $t {
            fn is_same_type(other: i32) -> bool {
                $(other == NPY_TYPES::$npy_types as i32 ||)+ false
            }
            fn npy_data_type() -> NpyDataType {
                NpyDataType::$npy_dat_t
            }
            fn typenum_default() -> i32 {
                let t = ($(NPY_TYPES::$npy_types, )+);
                t.0 as i32
            }
        }
    };
}

impl_type_num!(bool, Bool, NPY_BOOL);
impl_type_num!(i8, Int8, NPY_BYTE);
impl_type_num!(i16, Int16, NPY_SHORT);
impl_type_num!(u8, Uint8, NPY_UBYTE);
impl_type_num!(u16, Uint16, NPY_USHORT);
impl_type_num!(f32, Float32, NPY_FLOAT);
impl_type_num!(f64, Float64, NPY_DOUBLE);
impl_type_num!(c32, Complex32, NPY_CFLOAT);
impl_type_num!(c64, Complex64, NPY_CDOUBLE);

cfg_if! {
    if #[cfg(any(target_pointer_width = "32", windows))] {
        impl_type_num!(i32, Int32, NPY_INT, NPY_LONG);
        impl_type_num!(u32, Uint32, NPY_UINT, NPY_ULONG);
        impl_type_num!(i64, Int64, NPY_LONGLONG);
        impl_type_num!(u64, Uint64, NPY_ULONGLONG);
    } else if #[cfg(all(target_pointer_width = "64", not(windows)))] {
        impl_type_num!(i32, Int32, NPY_INT);
        impl_type_num!(u32, Uint32, NPY_UINT);
        impl_type_num!(i64, Int64, NPY_LONG, NPY_LONGLONG);
        impl_type_num!(u64, Uint64, NPY_LONG, NPY_ULONGLONG);
    }
}

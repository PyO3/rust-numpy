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
            x if x == NPY_TYPES::NPY_LONG as i32 => {
                if cfg!(any(windows, target_pointer_width = "32")) {
                    NpyDataType::Int32
                } else {
                    NpyDataType::Int64
                }
            }
            x if x == NPY_TYPES::NPY_LONGLONG as i32 => NpyDataType::Int64,
            x if x == NPY_TYPES::NPY_UBYTE as i32 => NpyDataType::Uint8,
            x if x == NPY_TYPES::NPY_USHORT as i32 => NpyDataType::Uint16,
            x if x == NPY_TYPES::NPY_UINT as i32 => NpyDataType::Uint32,
            x if x == NPY_TYPES::NPY_ULONG as i32 => {
                if cfg!(any(windows, target_pointer_width = "32")) {
                    NpyDataType::Uint32
                } else {
                    NpyDataType::Uint64
                }
            }
            x if x == NPY_TYPES::NPY_ULONGLONG as i32 => NpyDataType::Uint64,
            x if x == NPY_TYPES::NPY_FLOAT as i32 => NpyDataType::Float32,
            x if x == NPY_TYPES::NPY_DOUBLE as i32 => NpyDataType::Float64,
            x if x == NPY_TYPES::NPY_CFLOAT as i32 => NpyDataType::Complex32,
            x if x == NPY_TYPES::NPY_CDOUBLE as i32 => NpyDataType::Complex64,
            _ => NpyDataType::Unsupported,
        }
    }
}

pub trait TypeNum: Clone {
    fn typenum_enum() -> NPY_TYPES;
    fn typenum() -> i32 {
        Self::typenum_enum() as i32
    }
    fn to_npy_data_type(self) -> NpyDataType;
}

macro_rules! impl_type_num {
    ($t:ty, $npy_t:ident, $npy_dat_t:ident) => {
        impl TypeNum for $t {
            fn typenum_enum() -> NPY_TYPES {
                NPY_TYPES::$npy_t
            }
            fn to_npy_data_type(self) -> NpyDataType {
                NpyDataType::$npy_dat_t
            }
        }
    };
} // impl_type_num!

impl_type_num!(bool, NPY_BOOL, Bool);
impl_type_num!(i8, NPY_BYTE, Int8);
impl_type_num!(i16, NPY_SHORT, Int16);
impl_type_num!(i32, NPY_INT, Int32);
impl_type_num!(i64, NPY_LONGLONG, Int64);
impl_type_num!(u8, NPY_UBYTE, Uint8);
impl_type_num!(u16, NPY_USHORT, Uint16);
impl_type_num!(u32, NPY_UINT, Uint32);
impl_type_num!(u64, NPY_ULONGLONG, Uint64);
impl_type_num!(f32, NPY_FLOAT, Float32);
impl_type_num!(f64, NPY_DOUBLE, Float64);
impl_type_num!(c32, NPY_CFLOAT, Complex32);
impl_type_num!(c64, NPY_CDOUBLE, Complex64);


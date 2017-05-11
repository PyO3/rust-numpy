
pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;

pub use super::npyffi::NPY_TYPES;

pub trait TypeNum {
    fn typenum_enum() -> NPY_TYPES;
    fn typenum() -> i32 {
        Self::typenum_enum() as i32
    }
}

macro_rules! impl_type_num {
    ($t:ty, $npy_t:ident) => {
impl TypeNum for $t {
    fn typenum_enum() -> NPY_TYPES {
        NPY_TYPES::$npy_t
    }
}
}} // impl_type_num!

impl_type_num!(bool, NPY_BOOL);
impl_type_num!(i32, NPY_INT);
impl_type_num!(i64, NPY_LONG);
impl_type_num!(u32, NPY_UINT);
impl_type_num!(u64, NPY_ULONG);
impl_type_num!(f32, NPY_FLOAT);
impl_type_num!(f64, NPY_DOUBLE);
impl_type_num!(c32, NPY_CFLOAT);
impl_type_num!(c64, NPY_CDOUBLE);

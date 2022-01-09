use std::mem::size_of;
use std::os::raw::{
    c_char, c_int, c_long, c_longlong, c_short, c_uint, c_ulong, c_ulonglong, c_ushort,
};

use num_traits::{Bounded, Zero};
use pyo3::{
    ffi::{self, PyTuple_Size},
    prelude::*,
    pyobject_native_type_core,
    types::{PyDict, PyTuple, PyType},
    AsPyPointer, FromPyObject, FromPyPointer, PyNativeType,
};

use crate::npyffi::{
    NpyTypes, PyArray_Descr, NPY_ALIGNED_STRUCT, NPY_BYTEORDER_CHAR, NPY_ITEM_HASOBJECT, NPY_TYPES,
    PY_ARRAY_API,
};

pub use num_complex::{Complex32, Complex64};
use pyo3::exceptions::{PyIndexError, PyValueError};

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
///     assert!(dtype.is_equiv_to(numpy::dtype::<f64>(py)));
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

/// Returns the type descriptor ("dtype") for a registered type.
pub fn dtype<T: Element>(py: Python) -> &PyArrayDescr {
    T::get_dtype(py)
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

    /// Shortcut for creating a descriptor of 'object' type.
    pub fn object(py: Python) -> &Self {
        Self::from_npy_type(py, NPY_TYPES::NPY_OBJECT)
    }

    /// Returns the type descriptor ("dtype") for a registered type.
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

    /// Returns the
    /// [array scalar](https://numpy.org/doc/stable/reference/arrays.scalars.html)
    /// corresponding to this dtype.
    ///
    /// Equivalent to [`np.dtype.type`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.type.html).
    pub fn typeobj(&self) -> &PyType {
        let dtype_type_ptr = unsafe { *self.as_dtype_ptr() }.typeobj;
        unsafe { PyType::from_type_ptr(self.py(), dtype_type_ptr) }
    }

    #[doc(hidden)]
    #[deprecated(note = "`get_type()` is deprecated, please use `typeobj()` instead")]
    pub fn get_type(&self) -> &PyType {
        self.typeobj()
    }

    /// Returns a unique number for each of the 21 different built-in
    /// [enumerated types](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types).
    ///
    /// These are roughly ordered from least-to-most precision.
    ///
    /// Equivalent to [`np.dtype.num`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.num.html).
    pub fn num(&self) -> c_int {
        unsafe { *self.as_dtype_ptr() }.type_num
    }

    /// Returns the element size of this data-type object.
    ///
    /// Equivalent to [`np.dtype.itemsize`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.itemsize.html).
    pub fn itemsize(&self) -> usize {
        unsafe { *self.as_dtype_ptr() }.elsize.max(0) as _
    }

    /// Returns the required alignment (bytes) of this data-type according to the compiler.
    ///
    /// Equivalent to [`np.dtype.alignment`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.alignment.html).
    pub fn alignment(&self) -> usize {
        unsafe { *self.as_dtype_ptr() }.alignment.max(0) as _
    }

    /// Returns a character indicating the byte-order of this data-type object.
    ///
    /// All built-in data-type objects have byteorder either `=` or `|`.
    ///
    /// Equivalent to [`np.dtype.byteorder`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html).
    pub fn byteorder(&self) -> u8 {
        unsafe { *self.as_dtype_ptr() }.byteorder.max(0) as _
    }

    /// Returns a unique character code for each of the 21 different built-in types.
    ///
    /// Note: structured data types are categorized as `V` (void).
    ///
    /// Equivalent to [`np.dtype.char`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.char.html).
    pub fn char(&self) -> u8 {
        unsafe { *self.as_dtype_ptr() }.type_.max(0) as _
    }

    /// Returns a character code (one of `biufcmMOSUV`) identifying the general kind of data.
    ///
    /// Note: structured data types are categorized as `V` (void).
    ///
    /// Equivalent to [`np.dtype.kind`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html).
    pub fn kind(&self) -> u8 {
        unsafe { *self.as_dtype_ptr() }.kind.max(0) as _
    }

    /// Returns bit-flags describing how this data type is to be interpreted.
    ///
    /// Equivalent to [`np.dtype.flags`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.flags.html).
    pub fn flags(&self) -> c_char {
        unsafe { *self.as_dtype_ptr() }.flags
    }

    /// Returns the number of dimensions if this data type describes a sub-array, and `0` otherwise.
    ///
    /// Equivalent to [`np.dtype.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.ndim.html).
    pub fn ndim(&self) -> usize {
        if !self.has_subarray() {
            return 0;
        }
        unsafe { PyTuple_Size((*((*self.as_dtype_ptr()).subarray)).shape).max(0) as _ }
    }

    /// Returns dtype for the base element of subarrays, regardless of their dimension or shape.
    ///
    /// If the dtype is not a subarray, returns self.
    ///
    /// Equivalent to [`np.dtype.base`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.base.html).
    pub fn base(&self) -> &PyArrayDescr {
        if !self.has_subarray() {
            self
        } else {
            unsafe {
                Self::from_borrowed_ptr(self.py(), (*(*self.as_dtype_ptr()).subarray).base as _)
            }
        }
    }

    /// Returns the shape of the sub-array.
    ///
    /// If the dtype is not a sub-array, an empty vector is returned.
    ///
    /// Equivalent to [`np.dtype.shape`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.shape.html).
    pub fn shape(&self) -> Vec<usize> {
        if !self.has_subarray() {
            vec![]
        } else {
            // Panic-wise: numpy guarantees that shape is a tuple of non-negative integers
            unsafe {
                PyTuple::from_borrowed_ptr(self.py(), (*(*self.as_dtype_ptr()).subarray).shape)
            }
            .extract()
            .unwrap()
        }
    }

    /// Returns true if the dtype is a sub-array at the top level.
    ///
    /// Equivalent to [`np.dtype.hasobject`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.hasobject.html).
    pub fn has_object(&self) -> bool {
        self.flags() & NPY_ITEM_HASOBJECT != 0
    }

    /// Returns true if the dtype is a struct which maintains field alignment.
    ///
    /// This flag is sticky, so when combining multiple structs together, it is preserved
    /// and produces new dtypes which are also aligned.
    ///
    /// Equivalent to [`np.dtype.isalignedstruct`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.isalignedstruct.html).
    pub fn is_aligned_struct(&self) -> bool {
        self.flags() & NPY_ALIGNED_STRUCT != 0
    }

    /// Returns true if the data type is a sub-array.
    pub fn has_subarray(&self) -> bool {
        // equivalent to PyDataType_HASSUBARRAY(self)
        unsafe { !(*self.as_dtype_ptr()).subarray.is_null() }
    }

    /// Returns true if the data type is a structured type.
    pub fn has_fields(&self) -> bool {
        // equivalent to PyDataType_HASFIELDS(self)
        unsafe { !(*self.as_dtype_ptr()).names.is_null() }
    }

    /// Returns true if data type byteorder is native, or `None` if not applicable.
    pub fn is_native_byteorder(&self) -> Option<bool> {
        // based on PyArray_ISNBO(self->byteorder)
        match self.byteorder() {
            b'=' => Some(true),
            b'|' => None,
            byteorder if byteorder == NPY_BYTEORDER_CHAR::NPY_NATBYTE as u8 => Some(true),
            _ => Some(false),
        }
    }

    /// Returns an ordered list of field names, or `None` if there are no fields.
    ///
    /// The names are ordered according to increasing byte offset.
    ///
    /// Equivalent to [`np.dtype.names`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.names.html).
    pub fn names(&self) -> Option<Vec<&str>> {
        if !self.has_fields() {
            return None;
        }
        let names = unsafe { PyTuple::from_borrowed_ptr(self.py(), (*self.as_dtype_ptr()).names) };
        FromPyObject::extract(names).ok()
    }

    /// Returns the dtype and offset of a field with a given name.
    ///
    /// This method will return an error if the dtype is not structured, or if it doesn't
    /// contain a field with a given name.
    ///
    /// The list of all names can be found via [`PyArrayDescr::names`].
    ///
    /// Equivalent to retrieving a single item from
    /// [`np.dtype.fields`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.fields.html).
    pub fn get_field(&self, name: &str) -> PyResult<(&PyArrayDescr, usize)> {
        if !self.has_fields() {
            return Err(PyValueError::new_err(
                "cannot get field information: dtype has no fields",
            ));
        }
        let dict = unsafe { PyDict::from_borrowed_ptr(self.py(), (*self.as_dtype_ptr()).fields) };
        // Panic-wise: numpy guarantees that fields are tuples of proper size and type
        let tuple = dict
            .get_item(name)
            .ok_or_else(|| PyIndexError::new_err(name.to_owned()))?
            .downcast::<PyTuple>()
            .unwrap();
        // (note: we can't just extract the entire tuple since 3rd element can be a title)
        let dtype = FromPyObject::extract(tuple.as_ref().get_item(0).unwrap()).unwrap();
        let offset = FromPyObject::extract(tuple.as_ref().get_item(1).unwrap()).unwrap();
        Ok((dtype, offset))
    }
}

/// Represents that a type can be an element of `PyArray`.
///
/// Currently, only integer/float/complex/object types are supported.
/// If you come up with a nice implementation for some other types, we're happy to receive your PR :)
/// You may refer to the [numpy document](https://numpy.org/doc/stable/reference/c-api/dtype.html#enumerated-types)
/// for all types that numpy supports.
///
/// # Safety
///
/// A type `T` that implements this trait should be safe when managed in numpy
/// array, thus implementing this trait is marked unsafe. Data types that don't
/// contain Python objects (i.e., either the object type itself or record types
/// containing object-type fields) are assumed to be trivially copyable, which
/// is reflected in the `IS_COPY` flag. Furthermore, it is assumed that for
/// the object type the elements are pointers into the Python heap and that the
/// corresponding `Clone` implemenation will never panic as it only increases
/// the reference count.
///
/// # Custom element types
///
/// Note that we cannot safely store `Py<T>` where `T: PyClass`, because the type information would be
/// eliminated in the resulting NumPy array.
/// In other words, objects are always treated as `Py<PyAny>` (a.k.a. `PyObject`) by Python code,
/// and only `Py<PyAny>` can be stored in a type safe manner.
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
    ($ty:ty => $npy_type:ident $(,#[$meta:meta])*) => {
        impl_element_scalar!(@impl: $ty, NPY_TYPES::$npy_type $(,#[$meta])*);
    };
    ($($tys:ty),+) => {
        $(impl_element_scalar!(@impl: $tys, npy_int_type::<$tys>());)+
    };
}

impl_element_scalar!(bool => NPY_BOOL);
impl_element_scalar!(i8, i16, i32, i64);
impl_element_scalar!(u8, u16, u32, u64);
impl_element_scalar!(f32 => NPY_FLOAT);
impl_element_scalar!(f64 => NPY_DOUBLE);
impl_element_scalar!(Complex32 => NPY_CFLOAT,
    #[doc = "Complex type with `f32` components which maps to `np.csingle` (`np.complex64`)."]);
impl_element_scalar!(Complex64 => NPY_CDOUBLE,
    #[doc = "Complex type with `f64` components which maps to `np.cdouble` (`np.complex128`)."]);

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl_element_scalar!(usize, isize);

unsafe impl Element for PyObject {
    const IS_COPY: bool = false;

    fn get_dtype(py: Python) -> &PyArrayDescr {
        PyArrayDescr::object(py)
    }
}

#[cfg(test)]
mod tests {
    use std::os::raw::c_int;

    use pyo3::{py_run, types::PyDict, PyObject};

    use super::{dtype, Complex32, Complex64, Element, PyArrayDescr};
    use crate::npyffi::{NPY_ALIGNED_STRUCT, NPY_ITEM_HASOBJECT, NPY_NEEDS_PYAPI, NPY_TYPES};

    #[test]
    fn test_dtype_names() {
        fn type_name<T: Element>(py: pyo3::Python) -> &str {
            dtype::<T>(py).typeobj().name().unwrap()
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
            #[cfg(target_pointer_width = "32")]
            {
                assert_eq!(type_name::<usize>(py), "uint32");
                assert_eq!(type_name::<isize>(py), "int32");
            }
            #[cfg(target_pointer_width = "64")]
            {
                assert_eq!(type_name::<usize>(py), "uint64");
                assert_eq!(type_name::<isize>(py), "int64");
            }
        });
    }

    #[test]
    fn test_dtype_methods_scalar() {
        pyo3::Python::with_gil(|py| {
            let dt = dtype::<f64>(py);

            assert_eq!(dt.num(), NPY_TYPES::NPY_DOUBLE as c_int);
            assert_eq!(dt.flags(), 0);
            assert_eq!(dt.typeobj().name().unwrap(), "float64");
            assert_eq!(dt.char(), b'd');
            assert_eq!(dt.kind(), b'f');
            assert_eq!(dt.byteorder(), b'=');
            assert_eq!(dt.is_native_byteorder(), Some(true));
            assert_eq!(dt.itemsize(), 8);
            assert_eq!(dt.alignment(), 8);
            assert!(!dt.has_object());
            assert_eq!(dt.names(), None);
            assert!(!dt.has_fields());
            assert!(!dt.is_aligned_struct());
            assert!(!dt.has_subarray());
            assert!(dt.base().is_equiv_to(dt));
            assert_eq!(dt.ndim(), 0);
            assert_eq!(dt.shape(), vec![]);
        });
    }

    #[test]
    fn test_dtype_methods_subarray() {
        pyo3::Python::with_gil(|py| {
            let locals = PyDict::new(py);
            py_run!(
                py,
                *locals,
                "dtype = __import__('numpy').dtype(('f8', (2, 3)))"
            );
            let dt = locals
                .get_item("dtype")
                .unwrap()
                .downcast::<PyArrayDescr>()
                .unwrap();

            assert_eq!(dt.num(), NPY_TYPES::NPY_VOID as c_int);
            assert_eq!(dt.flags(), 0);
            assert_eq!(dt.typeobj().name().unwrap(), "void");
            assert_eq!(dt.char(), b'V');
            assert_eq!(dt.kind(), b'V');
            assert_eq!(dt.byteorder(), b'|');
            assert_eq!(dt.is_native_byteorder(), None);
            assert_eq!(dt.itemsize(), 48);
            assert_eq!(dt.alignment(), 8);
            assert!(!dt.has_object());
            assert_eq!(dt.names(), None);
            assert!(!dt.has_fields());
            assert!(!dt.is_aligned_struct());
            assert!(dt.has_subarray());
            assert_eq!(dt.ndim(), 2);
            assert_eq!(dt.shape(), vec![2, 3]);
            assert!(dt.base().is_equiv_to(dtype::<f64>(py)));
        });
    }

    #[test]
    fn test_dtype_methods_record() {
        pyo3::Python::with_gil(|py| {
            let locals = PyDict::new(py);
            py_run!(
                py,
                *locals,
                "dtype = __import__('numpy').dtype([('x', 'u1'), ('y', 'f8'), ('z', 'O')], align=True)"
            );
            let dt = locals
                .get_item("dtype")
                .unwrap()
                .downcast::<PyArrayDescr>()
                .unwrap();

            assert_eq!(dt.num(), NPY_TYPES::NPY_VOID as c_int);
            assert_ne!(dt.flags() & NPY_ITEM_HASOBJECT, 0);
            assert_ne!(dt.flags() & NPY_NEEDS_PYAPI, 0);
            assert_ne!(dt.flags() & NPY_ALIGNED_STRUCT, 0);
            assert_eq!(dt.typeobj().name().unwrap(), "void");
            assert_eq!(dt.char(), b'V');
            assert_eq!(dt.kind(), b'V');
            assert_eq!(dt.byteorder(), b'|');
            assert_eq!(dt.is_native_byteorder(), None);
            assert_eq!(dt.itemsize(), 24);
            assert_eq!(dt.alignment(), 8);
            assert!(dt.has_object());
            assert_eq!(dt.names(), Some(vec!["x", "y", "z"]));
            assert!(dt.has_fields());
            assert!(dt.is_aligned_struct());
            assert!(!dt.has_subarray());
            assert_eq!(dt.ndim(), 0);
            assert_eq!(dt.shape(), vec![]);
            assert!(dt.base().is_equiv_to(dt));
            let x = dt.get_field("x").unwrap();
            assert!(x.0.is_equiv_to(dtype::<u8>(py)));
            assert_eq!(x.1, 0);
            let y = dt.get_field("y").unwrap();
            assert!(y.0.is_equiv_to(dtype::<f64>(py)));
            assert_eq!(y.1, 8);
            let z = dt.get_field("z").unwrap();
            assert!(z.0.is_equiv_to(dtype::<PyObject>(py)));
            assert_eq!(z.1, 16);
        });
    }
}

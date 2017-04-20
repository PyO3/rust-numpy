
extern crate libc;
extern crate python3_sys as pyffi;

use pyffi::*;
use libc::FILE;

pub type npy_intp = Py_intptr_t;
pub type npy_hash_t = Py_hash_t;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum NPY_CLIPMODE {
    NPY_CLIP = 0,
    NPY_WRAP = 1,
    NPY_RAISE = 2,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct _object {
    pub ob_refcnt: Py_ssize_t,
    pub ob_type: *mut PyTypeObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyMethodDef {
    pub ml_name: *const ::std::os::raw::c_char,
    pub ml_meth: PyCFunction,
    pub ml_flags: ::std::os::raw::c_int,
    pub ml_doc: *const ::std::os::raw::c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_Descr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: ::std::os::raw::c_char,
    pub type_: ::std::os::raw::c_char,
    pub byteorder: ::std::os::raw::c_char,
    pub flags: ::std::os::raw::c_char,
    pub type_num: ::std::os::raw::c_int,
    pub elsize: ::std::os::raw::c_int,
    pub alignment: ::std::os::raw::c_int,
    pub subarray: *mut _PyArray_Descr__arr_descr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub f: *mut PyArray_ArrFuncs,
    pub metadata: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
    pub hash: npy_hash_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyMemberDef([u8; 0]);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyTypeObject {
    pub ob_base: PyVarObject,
    pub tp_name: *const ::std::os::raw::c_char,
    pub tp_basicsize: Py_ssize_t,
    pub tp_itemsize: Py_ssize_t,
    pub tp_dealloc: destructor,
    pub tp_print: printfunc,
    pub tp_getattr: getattrfunc,
    pub tp_setattr: setattrfunc,
    pub tp_as_async: *mut PyAsyncMethods,
    pub tp_repr: reprfunc,
    pub tp_as_number: *mut PyNumberMethods,
    pub tp_as_sequence: *mut PySequenceMethods,
    pub tp_as_mapping: *mut PyMappingMethods,
    pub tp_hash: hashfunc,
    pub tp_call: ternaryfunc,
    pub tp_str: reprfunc,
    pub tp_getattro: getattrofunc,
    pub tp_setattro: setattrofunc,
    pub tp_as_buffer: *mut PyBufferProcs,
    pub tp_flags: ::std::os::raw::c_ulong,
    pub tp_doc: *const ::std::os::raw::c_char,
    pub tp_traverse: traverseproc,
    pub tp_clear: inquiry,
    pub tp_richcompare: richcmpfunc,
    pub tp_weaklistoffset: Py_ssize_t,
    pub tp_iter: getiterfunc,
    pub tp_iternext: iternextfunc,
    pub tp_methods: *mut PyMethodDef,
    pub tp_members: *mut PyMemberDef,
    pub tp_getset: *mut PyGetSetDef,
    pub tp_base: *mut PyTypeObject,
    pub tp_dict: *mut PyObject,
    pub tp_descr_get: descrgetfunc,
    pub tp_descr_set: descrsetfunc,
    pub tp_dictoffset: Py_ssize_t,
    pub tp_init: initproc,
    pub tp_alloc: allocfunc,
    pub tp_new: newfunc,
    pub tp_free: freefunc,
    pub tp_is_gc: inquiry,
    pub tp_bases: *mut PyObject,
    pub tp_mro: *mut PyObject,
    pub tp_cache: *mut PyObject,
    pub tp_subclasses: *mut PyObject,
    pub tp_weaklist: *mut PyObject,
    pub tp_del: destructor,
    pub tp_version_tag: ::std::os::raw::c_uint,
    pub tp_finalize: destructor,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrayObject {
    pub ob_base: PyObject,
    pub data: *mut ::std::os::raw::c_char,
    pub nd: ::std::os::raw::c_int,
    pub dimensions: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub base: *mut PyObject,
    pub descr: *mut PyArray_Descr,
    pub flags: ::std::os::raw::c_int,
    pub weakreflist: *mut PyObject,
}


#[repr(C)]
#[derive(Copy, Clone)]
pub struct _PyArray_Descr__arr_descr {
    pub base: *mut PyArray_Descr,
    pub shape: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_ArrFuncs {
    pub cast: [PyArray_VectorUnaryFunc; 21usize],
    pub getitem: PyArray_GetItemFunc,
    pub setitem: PyArray_SetItemFunc,
    pub copyswapn: PyArray_CopySwapNFunc,
    pub copyswap: PyArray_CopySwapFunc,
    pub compare: PyArray_CompareFunc,
    pub argmax: PyArray_ArgFunc,
    pub dotfunc: PyArray_DotFunc,
    pub scanfunc: PyArray_ScanFunc,
    pub fromstr: PyArray_FromStrFunc,
    pub nonzero: PyArray_NonzeroFunc,
    pub fill: PyArray_FillFunc,
    pub fillwithscalar: PyArray_FillWithScalarFunc,
    pub sort: [PyArray_SortFunc; 3usize],
    pub argsort: [PyArray_ArgSortFunc; 3usize],
    pub castdict: *mut PyObject,
    pub scalarkind: PyArray_ScalarKindFunc,
    pub cancastscalarkindto: *mut *mut ::std::os::raw::c_int,
    pub cancastto: *mut ::std::os::raw::c_int,
    pub fastclip: PyArray_FastClipFunc,
    pub fastputmask: PyArray_FastPutmaskFunc,
    pub fasttake: PyArray_FastTakeFunc,
    pub argmin: PyArray_ArgFunc,
}

pub type NpyAuxData = NpyAuxData_tag;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NpyAuxData_tag {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut ::std::os::raw::c_void; 2usize],
}

pub type PyArray_GetItemFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void)
                                                 -> *mut _object>;
pub type PyArray_SetItemFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut PyObject,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_CopySwapNFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: npy_intp,
                                                 arg5: npy_intp,
                                                 arg6: ::std::os::raw::c_int,
                                                 arg7: *mut ::std::os::raw::c_void)>;
pub type PyArray_CopySwapFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: ::std::os::raw::c_int,
                                                 arg4: *mut ::std::os::raw::c_void)>;
pub type PyArray_NonzeroFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_uchar>;
pub type PyArray_CompareFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                                 arg2: *const ::std::os::raw::c_void,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_DotFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: npy_intp,
                                                 arg5: *mut ::std::os::raw::c_void,
                                                 arg6: npy_intp,
                                                 arg7: *mut ::std::os::raw::c_void)>;
pub type PyArray_VectorUnaryFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut ::std::os::raw::c_void,
                                                 arg3: npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void,
                                                 arg5: *mut ::std::os::raw::c_void)>;
pub type PyArray_ScanFunc =
    ::std::option::Option<unsafe extern "C" fn(fp: *mut FILE,
                                                 dptr: *mut ::std::os::raw::c_void,
                                                 ignore: *mut ::std::os::raw::c_char,
                                                 arg1: *mut PyArray_Descr)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FromStrFunc =
    ::std::option::Option<unsafe extern "C" fn(s: *mut ::std::os::raw::c_char,
                                                 dptr: *mut ::std::os::raw::c_void,
                                                 endptr: *mut *mut ::std::os::raw::c_char,
                                                 arg1: *mut PyArray_Descr)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FillFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_SortFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgSortFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_PartitionFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: *mut npy_intp,
                                                 arg5: *mut npy_intp,
                                                 arg6: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ArgPartitionFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: *mut npy_intp,
                                                 arg3: npy_intp,
                                                 arg4: npy_intp,
                                                 arg5: *mut npy_intp,
                                                 arg6: *mut npy_intp,
                                                 arg7: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FillWithScalarFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                                                 arg2: npy_intp,
                                                 arg3: *mut ::std::os::raw::c_void,
                                                 arg4: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_ScalarKindFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void)
                                                 -> ::std::os::raw::c_int>;
pub type PyArray_FastClipFunc =
    ::std::option::Option<unsafe extern "C" fn(in_: *mut ::std::os::raw::c_void,
                                                 n_in: npy_intp,
                                                 min: *mut ::std::os::raw::c_void,
                                                 max: *mut ::std::os::raw::c_void,
                                                 out: *mut ::std::os::raw::c_void)>;
pub type PyArray_FastPutmaskFunc =
    ::std::option::Option<unsafe extern "C" fn(in_: *mut ::std::os::raw::c_void,
                                                 mask: *mut ::std::os::raw::c_void,
                                                 n_in: npy_intp,
                                                 values: *mut ::std::os::raw::c_void,
                                                 nv: npy_intp)>;
pub type PyArray_FastTakeFunc =
    ::std::option::Option<unsafe extern "C" fn(dest: *mut ::std::os::raw::c_void,
                                                 src: *mut ::std::os::raw::c_void,
                                                 indarray: *mut npy_intp,
                                                 nindarray: npy_intp,
                                                 n_outer: npy_intp,
                                                 m_middle: npy_intp,
                                                 nelem: npy_intp,
                                                 clipmode: NPY_CLIPMODE)
                                                 -> ::std::os::raw::c_int>;
pub type NpyAuxData_FreeFunc = ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData)>;
pub type NpyAuxData_CloneFunc =
    ::std::option::Option<unsafe extern "C" fn(arg1: *mut NpyAuxData) -> *mut NpyAuxData_tag>;

use super::{npy_char, npy_uint32};
use std::os::raw::c_int;

pub const NPY_ARRAY_C_CONTIGUOUS: c_int = 0x0001;
pub const NPY_ARRAY_F_CONTIGUOUS: c_int = 0x0002;
pub const NPY_ARRAY_OWNDATA: c_int = 0x0004;
pub const NPY_ARRAY_FORCECAST: c_int = 0x0010;
pub const NPY_ARRAY_ENSURECOPY: c_int = 0x0020;
pub const NPY_ARRAY_ENSUREARRAY: c_int = 0x0040;
pub const NPY_ARRAY_ELEMENTSTRIDES: c_int = 0x0080;
pub const NPY_ARRAY_ALIGNED: c_int = 0x0100;
pub const NPY_ARRAY_NOTSWAPPED: c_int = 0x0200;
pub const NPY_ARRAY_WRITEABLE: c_int = 0x0400;
pub const NPY_ARRAY_UPDATEIFCOPY: c_int = 0x1000;
pub const NPY_ARRAY_WRITEBACKIFCOPY: c_int = 0x2000;
pub const NPY_ARRAY_BEHAVED: c_int = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
pub const NPY_ARRAY_BEHAVED_NS: c_int = NPY_ARRAY_BEHAVED | NPY_ARRAY_NOTSWAPPED;
pub const NPY_ARRAY_CARRAY: c_int = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED;
pub const NPY_ARRAY_CARRAY_RO: c_int = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
pub const NPY_ARRAY_FARRAY: c_int = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED;
pub const NPY_ARRAY_FARRAY_RO: c_int = NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED;
pub const NPY_ARRAY_DEFAULT: c_int = NPY_ARRAY_CARRAY;
pub const NPY_ARRAY_IN_ARRAY: c_int = NPY_ARRAY_CARRAY_RO;
pub const NPY_ARRAY_OUT_ARRAY: c_int = NPY_ARRAY_CARRAY;
pub const NPY_ARRAY_INOUT_ARRAY: c_int = NPY_ARRAY_CARRAY | NPY_ARRAY_UPDATEIFCOPY;
pub const NPY_ARRAY_INOUT_ARRAY2: c_int = NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY;
pub const NPY_ARRAY_IN_FARRAY: c_int = NPY_ARRAY_FARRAY_RO;
pub const NPY_ARRAY_OUT_FARRAY: c_int = NPY_ARRAY_FARRAY;
pub const NPY_ARRAY_INOUT_FARRAY: c_int = NPY_ARRAY_FARRAY | NPY_ARRAY_UPDATEIFCOPY;
pub const NPY_ARRAY_INOUT_FARRAY2: c_int = NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEBACKIFCOPY;
pub const NPY_ARRAY_UPDATE_ALL: c_int = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS;

pub const NPY_ITER_C_INDEX: npy_uint32 = 0x00000001;
pub const NPY_ITER_F_INDEX: npy_uint32 = 0x00000002;
pub const NPY_ITER_MULTI_INDEX: npy_uint32 = 0x00000004;
pub const NPY_ITER_EXTERNAL_LOOP: npy_uint32 = 0x00000008;
pub const NPY_ITER_COMMON_DTYPE: npy_uint32 = 0x00000010;
pub const NPY_ITER_REFS_OK: npy_uint32 = 0x00000020;
pub const NPY_ITER_ZEROSIZE_OK: npy_uint32 = 0x00000040;
pub const NPY_ITER_REDUCE_OK: npy_uint32 = 0x00000080;
pub const NPY_ITER_RANGED: npy_uint32 = 0x00000100;
pub const NPY_ITER_BUFFERED: npy_uint32 = 0x00000200;
pub const NPY_ITER_GROWINNER: npy_uint32 = 0x00000400;
pub const NPY_ITER_DELAY_BUFALLOC: npy_uint32 = 0x00000800;
pub const NPY_ITER_DONT_NEGATE_STRIDES: npy_uint32 = 0x00001000;
pub const NPY_ITER_COPY_IF_OVERLAP: npy_uint32 = 0x00002000;
pub const NPY_ITER_READWRITE: npy_uint32 = 0x00010000;
pub const NPY_ITER_READONLY: npy_uint32 = 0x00020000;
pub const NPY_ITER_WRITEONLY: npy_uint32 = 0x00040000;
pub const NPY_ITER_NBO: npy_uint32 = 0x00080000;
pub const NPY_ITER_ALIGNED: npy_uint32 = 0x00100000;
pub const NPY_ITER_CONTIG: npy_uint32 = 0x00200000;
pub const NPY_ITER_COPY: npy_uint32 = 0x00400000;
pub const NPY_ITER_UPDATEIFCOPY: npy_uint32 = 0x00800000;
pub const NPY_ITER_ALLOCATE: npy_uint32 = 0x01000000;
pub const NPY_ITER_NO_SUBTYPE: npy_uint32 = 0x02000000;
pub const NPY_ITER_VIRTUAL: npy_uint32 = 0x04000000;
pub const NPY_ITER_NO_BROADCAST: npy_uint32 = 0x08000000;
pub const NPY_ITER_WRITEMASKED: npy_uint32 = 0x10000000;
pub const NPY_ITER_ARRAYMASK: npy_uint32 = 0x20000000;
pub const NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE: npy_uint32 = 0x40000000;

pub const NPY_ITER_GLOBAL_FLAGS: npy_uint32 = 0x0000ffff;
pub const NPY_ITER_PER_OP_FLAGS: npy_uint32 = 0xffff0000;

pub const NPY_ITEM_REFCOUNT: npy_char = 0x01;
pub const NPY_ITEM_HASOBJECT: npy_char = 0x01;
pub const NPY_LIST_PICKLE: npy_char = 0x02;
pub const NPY_ITEM_IS_POINTER: npy_char = 0x04;
pub const NPY_NEEDS_INIT: npy_char = 0x08;
pub const NPY_NEEDS_PYAPI: npy_char = 0x10;
pub const NPY_USE_GETITEM: npy_char = 0x20;
pub const NPY_USE_SETITEM: npy_char = 0x40;
#[allow(overflowing_literals)]
pub const NPY_ALIGNED_STRUCT: npy_char = 0x80;
pub const NPY_FROM_FIELDS: npy_char =
    NPY_NEEDS_INIT | NPY_LIST_PICKLE | NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI;
pub const NPY_OBJECT_DTYPE_FLAGS: npy_char = NPY_LIST_PICKLE
    | NPY_USE_GETITEM
    | NPY_ITEM_IS_POINTER
    | NPY_ITEM_REFCOUNT
    | NPY_NEEDS_INIT
    | NPY_NEEDS_PYAPI;

pub const NPY_UFUNC_ZERO: c_int = 0;
pub const NPY_UFUNC_ONE: c_int = 1;
pub const NPY_UFUNC_MINUS_ONE: c_int = 2;
pub const NPY_UFUNC_NONE: c_int = -1;
pub const NPY_UFUNC_REORDERABLE_NONE: c_int = -2;
pub const NPY_UFUNC_IDENTITY_VALUE: c_int = -3;

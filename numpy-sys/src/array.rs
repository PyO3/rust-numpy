#![allow(non_camel_case_types)]

use libc::FILE;
use pyffi::*;
use std::os::raw::*;
use std::option::Option;

use super::types::*;
use super::iterator::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayFlagsObject {
    pub ob_base: PyObject,
    pub arr: *mut PyObject,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Dims {
    pub ptr: *mut npy_intp,
    pub len: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArray_Chunk {
    pub ob_base: PyObject,
    pub base: *mut PyObject,
    pub ptr: *mut c_void,
    pub len: npy_intp,
    pub flags: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PyArrayInterface {
    pub two: c_int,
    pub nd: c_int,
    pub typekind: c_char,
    pub itemsize: c_int,
    pub flags: c_int,
    pub shape: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub data: *mut c_void,
    pub descr: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArray_Descr {
    pub ob_base: PyObject,
    pub typeobj: *mut PyTypeObject,
    pub kind: c_char,
    pub type_: c_char,
    pub byteorder: c_char,
    pub flags: c_char,
    pub type_num: c_int,
    pub elsize: c_int,
    pub alignment: c_int,
    pub subarray: *mut PyArrray_ArrayDescr,
    pub fields: *mut PyObject,
    pub names: *mut PyObject,
    pub f: *mut PyArray_ArrFuncs,
    pub metadata: *mut PyObject,
    pub c_metadata: *mut NpyAuxData,
    pub hash: npy_hash_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrayObject {
    pub ob_base: PyObject,
    pub data: *mut c_char,
    pub nd: c_int,
    pub dimensions: *mut npy_intp,
    pub strides: *mut npy_intp,
    pub base: *mut PyObject,
    pub descr: *mut PyArray_Descr,
    pub flags: c_int,
    pub weakreflist: *mut PyObject,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PyArrray_ArrayDescr {
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
    pub cancastscalarkindto: *mut *mut c_int,
    pub cancastto: *mut c_int,
    pub fastclip: PyArray_FastClipFunc,
    pub fastputmask: PyArray_FastPutmaskFunc,
    pub fasttake: PyArray_FastTakeFunc,
    pub argmin: PyArray_ArgFunc,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct NpyAuxData {
    pub free: NpyAuxData_FreeFunc,
    pub clone: NpyAuxData_CloneFunc,
    pub reserved: [*mut c_void; 2usize],
}

pub type NpyAuxData_FreeFunc = Option<unsafe extern "C" fn(arg1: *mut NpyAuxData)>;

pub type NpyAuxData_CloneFunc = Option<unsafe extern "C" fn(arg1: *mut NpyAuxData) -> *mut NpyAuxData>;


pub type PyArray_GetItemFunc = Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut c_void) -> *mut PyObject>;
pub type PyArray_SetItemFunc = Option<unsafe extern "C" fn(arg1: *mut PyObject, arg2: *mut c_void, arg3: *mut c_void) -> c_int>;
pub type PyArray_CopySwapNFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                             arg2: npy_intp,
                                                             arg3: *mut c_void,
                                                             arg4: npy_intp,
                                                             arg5: npy_intp,
                                                             arg6: c_int,
                                                             arg7: *mut c_void)>;
pub type PyArray_CopySwapFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                            arg2: *mut c_void,
                                                            arg3: c_int,
                                                            arg4: *mut c_void)>;
pub type PyArray_NonzeroFunc = Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut c_void) -> c_uchar>;
pub type PyArray_CompareFunc = Option<unsafe extern "C" fn(arg1: *const c_void, arg2: *const c_void, arg3: *mut c_void) -> c_int>;
pub type PyArray_ArgFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                       arg2: npy_intp,
                                                       arg3: *mut npy_intp,
                                                       arg4: *mut c_void)
                                                       -> c_int>;
pub type PyArray_DotFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                       arg2: npy_intp,
                                                       arg3: *mut c_void,
                                                       arg4: npy_intp,
                                                       arg5: *mut c_void,
                                                       arg6: npy_intp,
                                                       arg7: *mut c_void)>;
pub type PyArray_VectorUnaryFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                               arg2: *mut c_void,
                                                               arg3: npy_intp,
                                                               arg4: *mut c_void,
                                                               arg5: *mut c_void)>;
pub type PyArray_ScanFunc = Option<unsafe extern "C" fn(fp: *mut FILE,
                                                        dptr: *mut c_void,
                                                        ignore: *mut c_char,
                                                        arg1: *mut PyArray_Descr)
                                                        -> c_int>;
pub type PyArray_FromStrFunc = Option<unsafe extern "C" fn(s: *mut c_char,
                                                           dptr: *mut c_void,
                                                           endptr: *mut *mut c_char,
                                                           arg1: *mut PyArray_Descr)
                                                           -> c_int>;
pub type PyArray_FillFunc = Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: npy_intp, arg3: *mut c_void) -> c_int>;
pub type PyArray_SortFunc = Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: npy_intp, arg3: *mut c_void) -> c_int>;
pub type PyArray_ArgSortFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                           arg2: *mut npy_intp,
                                                           arg3: npy_intp,
                                                           arg4: *mut c_void)
                                                           -> c_int>;
pub type PyArray_PartitionFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                             arg2: npy_intp,
                                                             arg3: npy_intp,
                                                             arg4: *mut npy_intp,
                                                             arg5: *mut npy_intp,
                                                             arg6: *mut c_void)
                                                             -> c_int>;
pub type PyArray_ArgPartitionFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                                arg2: *mut npy_intp,
                                                                arg3: npy_intp,
                                                                arg4: npy_intp,
                                                                arg5: *mut npy_intp,
                                                                arg6: *mut npy_intp,
                                                                arg7: *mut c_void)
                                                                -> c_int>;
pub type PyArray_FillWithScalarFunc = Option<unsafe extern "C" fn(arg1: *mut c_void,
                                                                  arg2: npy_intp,
                                                                  arg3: *mut c_void,
                                                                  arg4: *mut c_void)
                                                                  -> c_int>;
pub type PyArray_ScalarKindFunc = Option<unsafe extern "C" fn(arg1: *mut c_void) -> c_int>;
pub type PyArray_FastClipFunc = Option<unsafe extern "C" fn(in_: *mut c_void,
                                                            n_in: npy_intp,
                                                            min: *mut c_void,
                                                            max: *mut c_void,
                                                            out: *mut c_void)>;
pub type PyArray_FastPutmaskFunc = Option<unsafe extern "C" fn(in_: *mut c_void,
                                                               mask: *mut c_void,
                                                               n_in: npy_intp,
                                                               values: *mut c_void,
                                                               nv: npy_intp)>;
pub type PyArray_FastTakeFunc = Option<unsafe extern "C" fn(dest: *mut c_void,
                                                            src: *mut c_void,
                                                            indarray: *mut npy_intp,
                                                            nindarray: npy_intp,
                                                            n_outer: npy_intp,
                                                            m_middle: npy_intp,
                                                            nelem: npy_intp,
                                                            clipmode: NPY_CLIPMODE)
                                                            -> c_int>;

extern "C" {
    pub fn PyArray_GetNDArrayCVersion() -> c_uint;
    pub fn PyArray_SetNumericOps(dict: *mut PyObject) -> c_int;
    pub fn PyArray_GetNumericOps() -> *mut PyObject;
    pub fn PyArray_INCREF(mp: *mut PyArrayObject) -> c_int;
    pub fn PyArray_XDECREF(mp: *mut PyArrayObject) -> c_int;
    pub fn PyArray_SetStringFunction(op: *mut PyObject, repr: c_int);
    pub fn PyArray_DescrFromType(type_: c_int) -> *mut PyArray_Descr;
    pub fn PyArray_TypeObjectFromType(type_: c_int) -> *mut PyObject;
    pub fn PyArray_Zero(arr: *mut PyArrayObject) -> *mut c_char;
    pub fn PyArray_One(arr: *mut PyArrayObject) -> *mut c_char;
    pub fn PyArray_CastToType(arr: *mut PyArrayObject, dtype: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject;
    pub fn PyArray_CastTo(out: *mut PyArrayObject, mp: *mut PyArrayObject) -> c_int;
    pub fn PyArray_CastAnyTo(out: *mut PyArrayObject, mp: *mut PyArrayObject) -> c_int;
    pub fn PyArray_CanCastSafely(fromtype: c_int, totype: c_int) -> c_int;
    pub fn PyArray_CanCastTo(from: *mut PyArray_Descr, to: *mut PyArray_Descr) -> npy_bool;
    pub fn PyArray_ObjectType(op: *mut PyObject, minimum_type: c_int) -> c_int;
    pub fn PyArray_DescrFromObject(op: *mut PyObject, mintype: *mut PyArray_Descr) -> *mut PyArray_Descr;
    pub fn PyArray_ConvertToCommonType(op: *mut PyObject, retn: *mut c_int) -> *mut *mut PyArrayObject;
    pub fn PyArray_DescrFromScalar(sc: *mut PyObject) -> *mut PyArray_Descr;
    pub fn PyArray_DescrFromTypeObject(type_: *mut PyObject) -> *mut PyArray_Descr;
    pub fn PyArray_Size(op: *mut PyObject) -> npy_intp;
    pub fn PyArray_Scalar(data: *mut c_void, descr: *mut PyArray_Descr, base: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_FromScalar(scalar: *mut PyObject, outcode: *mut PyArray_Descr) -> *mut PyObject;
    pub fn PyArray_ScalarAsCtype(scalar: *mut PyObject, ctypeptr: *mut c_void);
    pub fn PyArray_CastScalarToCtype(scalar: *mut PyObject, ctypeptr: *mut c_void, outcode: *mut PyArray_Descr) -> c_int;
    pub fn PyArray_CastScalarDirect(scalar: *mut PyObject, indescr: *mut PyArray_Descr, ctypeptr: *mut c_void, outtype: c_int) -> c_int;
    pub fn PyArray_ScalarFromObject(object: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_GetCastFunc(descr: *mut PyArray_Descr, type_num: c_int) -> PyArray_VectorUnaryFunc;
    pub fn PyArray_FromDims(nd: c_int, d: *mut c_int, type_: c_int) -> *mut PyObject;
    pub fn PyArray_FromDimsAndDataAndDescr(nd: c_int, d: *mut c_int, descr: *mut PyArray_Descr, data: *mut c_char) -> *mut PyObject;
    pub fn PyArray_FromAny(op: *mut PyObject,
                           newtype: *mut PyArray_Descr,
                           min_depth: c_int,
                           max_depth: c_int,
                           flags: c_int,
                           context: *mut PyObject)
                           -> *mut PyObject;
    pub fn PyArray_EnsureArray(op: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_EnsureAnyArray(op: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_FromFile(fp: *mut FILE, dtype: *mut PyArray_Descr, num: npy_intp, sep: *mut c_char) -> *mut PyObject;
    pub fn PyArray_FromString(data: *mut c_char, slen: npy_intp, dtype: *mut PyArray_Descr, num: npy_intp, sep: *mut c_char) -> *mut PyObject;
    pub fn PyArray_FromBuffer(buf: *mut PyObject, type_: *mut PyArray_Descr, count: npy_intp, offset: npy_intp) -> *mut PyObject;
    pub fn PyArray_FromIter(obj: *mut PyObject, dtype: *mut PyArray_Descr, count: npy_intp) -> *mut PyObject;
    pub fn PyArray_Return(mp: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_GetField(self_: *mut PyArrayObject, typed: *mut PyArray_Descr, offset: c_int) -> *mut PyObject;
    pub fn PyArray_SetField(self_: *mut PyArrayObject, dtype: *mut PyArray_Descr, offset: c_int, val: *mut PyObject) -> c_int;
    pub fn PyArray_Byteswap(self_: *mut PyArrayObject, inplace: npy_bool) -> *mut PyObject;
    pub fn PyArray_Resize(self_: *mut PyArrayObject, newshape: *mut PyArray_Dims, refcheck: c_int, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_MoveInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int;
    pub fn PyArray_CopyInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int;
    pub fn PyArray_CopyAnyInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int;
    pub fn PyArray_CopyObject(dest: *mut PyArrayObject, src_object: *mut PyObject) -> c_int;
    pub fn PyArray_NewCopy(obj: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_ToList(self_: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_ToString(self_: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_ToFile(self_: *mut PyArrayObject, fp: *mut FILE, sep: *mut c_char, format: *mut c_char) -> c_int;
    pub fn PyArray_Dump(self_: *mut PyObject, file: *mut PyObject, protocol: c_int) -> c_int;
    pub fn PyArray_Dumps(self_: *mut PyObject, protocol: c_int) -> *mut PyObject;
    pub fn PyArray_ValidType(type_: c_int) -> c_int;
    pub fn PyArray_UpdateFlags(ret: *mut PyArrayObject, flagmask: c_int);
    pub fn PyArray_New(subtype: *mut PyTypeObject,
                       nd: c_int,
                       dims: *mut npy_intp,
                       type_num: c_int,
                       strides: *mut npy_intp,
                       data: *mut c_void,
                       itemsize: c_int,
                       flags: c_int,
                       obj: *mut PyObject)
                       -> *mut PyObject;
    pub fn PyArray_NewFromDescr(subtype: *mut PyTypeObject,
                                descr: *mut PyArray_Descr,
                                nd: c_int,
                                dims: *mut npy_intp,
                                strides: *mut npy_intp,
                                data: *mut c_void,
                                flags: c_int,
                                obj: *mut PyObject)
                                -> *mut PyObject;
    pub fn PyArray_DescrNew(base: *mut PyArray_Descr) -> *mut PyArray_Descr;
    pub fn PyArray_DescrNewFromType(type_num: c_int) -> *mut PyArray_Descr;
    pub fn PyArray_GetPriority(obj: *mut PyObject, default_: f64) -> f64;
    pub fn PyArray_IterNew(obj: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_MultiIterNew(n: c_int, ...) -> *mut PyObject;
    pub fn PyArray_PyIntAsInt(o: *mut PyObject) -> c_int;
    pub fn PyArray_PyIntAsIntp(o: *mut PyObject) -> npy_intp;
    pub fn PyArray_Broadcast(mit: *mut PyArrayMultiIterObject) -> c_int;
    pub fn PyArray_FillObjectArray(arr: *mut PyArrayObject, obj: *mut PyObject);
    pub fn PyArray_FillWithScalar(arr: *mut PyArrayObject, obj: *mut PyObject) -> c_int;
    pub fn PyArray_CheckStrides(elsize: c_int, nd: c_int, numbytes: npy_intp, offset: npy_intp, dims: *mut npy_intp, newstrides: *mut npy_intp) -> npy_bool;
    pub fn PyArray_DescrNewByteorder(self_: *mut PyArray_Descr, newendian: c_char) -> *mut PyArray_Descr;
    pub fn PyArray_IterAllButAxis(obj: *mut PyObject, inaxis: *mut c_int) -> *mut PyObject;
    pub fn PyArray_CheckFromAny(op: *mut PyObject,
                                descr: *mut PyArray_Descr,
                                min_depth: c_int,
                                max_depth: c_int,
                                requires: c_int,
                                context: *mut PyObject)
                                -> *mut PyObject;
    pub fn PyArray_FromArray(arr: *mut PyArrayObject, newtype: *mut PyArray_Descr, flags: c_int) -> *mut PyObject;
    pub fn PyArray_FromInterface(origin: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_FromStructInterface(input: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_FromArrayAttr(op: *mut PyObject, typecode: *mut PyArray_Descr, context: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_ScalarKind(typenum: c_int, arr: *mut *mut PyArrayObject) -> NPY_SCALARKIND;
    pub fn PyArray_CanCoerceScalar(thistype: c_int, neededtype: c_int, scalar: NPY_SCALARKIND) -> c_int;
    pub fn PyArray_NewFlagsObject(obj: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_CanCastScalar(from: *mut PyTypeObject, to: *mut PyTypeObject) -> npy_bool;
    pub fn PyArray_CompareUCS4(s1: *mut npy_ucs4, s2: *mut npy_ucs4, len: usize) -> c_int;
    pub fn PyArray_RemoveSmallest(multi: *mut PyArrayMultiIterObject) -> c_int;
    pub fn PyArray_ElementStrides(obj: *mut PyObject) -> c_int;
    pub fn PyArray_Item_INCREF(data: *mut c_char, descr: *mut PyArray_Descr);
    pub fn PyArray_Item_XDECREF(data: *mut c_char, descr: *mut PyArray_Descr);
    pub fn PyArray_FieldNames(fields: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_Transpose(ap: *mut PyArrayObject, permute: *mut PyArray_Dims) -> *mut PyObject;
    pub fn PyArray_TakeFrom(self0: *mut PyArrayObject, indices0: *mut PyObject, axis: c_int, out: *mut PyArrayObject, clipmode: NPY_CLIPMODE) -> *mut PyObject;
    pub fn PyArray_PutTo(self_: *mut PyArrayObject, values0: *mut PyObject, indices0: *mut PyObject, clipmode: NPY_CLIPMODE) -> *mut PyObject;
    pub fn PyArray_PutMask(self_: *mut PyArrayObject, values0: *mut PyObject, mask0: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_Repeat(aop: *mut PyArrayObject, op: *mut PyObject, axis: c_int) -> *mut PyObject;
    pub fn PyArray_Choose(ip: *mut PyArrayObject, op: *mut PyObject, out: *mut PyArrayObject, clipmode: NPY_CLIPMODE) -> *mut PyObject;
    pub fn PyArray_Sort(op: *mut PyArrayObject, axis: c_int, which: NPY_SORTKIND) -> c_int;
    pub fn PyArray_ArgSort(op: *mut PyArrayObject, axis: c_int, which: NPY_SORTKIND) -> *mut PyObject;
    pub fn PyArray_SearchSorted(op1: *mut PyArrayObject, op2: *mut PyObject, side: NPY_SEARCHSIDE, perm: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_ArgMax(op: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_ArgMin(op: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Reshape(self_: *mut PyArrayObject, shape: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_Newshape(self_: *mut PyArrayObject, newdims: *mut PyArray_Dims, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_Squeeze(self_: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_View(self_: *mut PyArrayObject, type_: *mut PyArray_Descr, pytype: *mut PyTypeObject) -> *mut PyObject;
    pub fn PyArray_SwapAxes(ap: *mut PyArrayObject, a1: c_int, a2: c_int) -> *mut PyObject;
    pub fn PyArray_Max(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Min(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Ptp(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Mean(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Trace(self_: *mut PyArrayObject, offset: c_int, axis1: c_int, axis2: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Diagonal(self_: *mut PyArrayObject, offset: c_int, axis1: c_int, axis2: c_int) -> *mut PyObject;
    pub fn PyArray_Clip(self_: *mut PyArrayObject, min: *mut PyObject, max: *mut PyObject, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Conjugate(self_: *mut PyArrayObject, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Nonzero(self_: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Std(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject, variance: c_int) -> *mut PyObject;
    pub fn PyArray_Sum(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_CumSum(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Prod(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_CumProd(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_All(self_: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Any(self_: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Compress(self_: *mut PyArrayObject, condition: *mut PyObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_Flatten(a: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_Ravel(arr: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject;
    pub fn PyArray_MultiplyList(l1: *mut npy_intp, n: c_int) -> npy_intp;
    pub fn PyArray_MultiplyIntList(l1: *mut c_int, n: c_int) -> c_int;
    pub fn PyArray_GetPtr(obj: *mut PyArrayObject, ind: *mut npy_intp) -> *mut c_void;
    pub fn PyArray_CompareLists(l1: *mut npy_intp, l2: *mut npy_intp, n: c_int) -> c_int;
    pub fn PyArray_AsCArray(op: *mut *mut PyObject, ptr: *mut c_void, dims: *mut npy_intp, nd: c_int, typedescr: *mut PyArray_Descr) -> c_int;
    pub fn PyArray_As1D(op: *mut *mut PyObject, ptr: *mut *mut c_char, d1: *mut c_int, typecode: c_int) -> c_int;
    pub fn PyArray_As2D(op: *mut *mut PyObject, ptr: *mut *mut *mut c_char, d1: *mut c_int, d2: *mut c_int, typecode: c_int) -> c_int;
    pub fn PyArray_Free(op: *mut PyObject, ptr: *mut c_void) -> c_int;
    pub fn PyArray_Converter(object: *mut PyObject, address: *mut *mut PyObject) -> c_int;
    pub fn PyArray_IntpFromSequence(seq: *mut PyObject, vals: *mut npy_intp, maxvals: c_int) -> c_int;
    pub fn PyArray_Concatenate(op: *mut PyObject, axis: c_int) -> *mut PyObject;
    pub fn PyArray_InnerProduct(op1: *mut PyObject, op2: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_MatrixProduct(op1: *mut PyObject, op2: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_CopyAndTranspose(op: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_Correlate(op1: *mut PyObject, op2: *mut PyObject, mode: c_int) -> *mut PyObject;
    pub fn PyArray_TypestrConvert(itemsize: c_int, gentype: c_int) -> c_int;
    pub fn PyArray_DescrConverter(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int;
    pub fn PyArray_DescrConverter2(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int;
    pub fn PyArray_IntpConverter(obj: *mut PyObject, seq: *mut PyArray_Dims) -> c_int;
    pub fn PyArray_BufferConverter(obj: *mut PyObject, buf: *mut PyArray_Chunk) -> c_int;
    pub fn PyArray_AxisConverter(obj: *mut PyObject, axis: *mut c_int) -> c_int;
    pub fn PyArray_BoolConverter(object: *mut PyObject, val: *mut npy_bool) -> c_int;
    pub fn PyArray_ByteorderConverter(obj: *mut PyObject, endian: *mut c_char) -> c_int;
    pub fn PyArray_OrderConverter(object: *mut PyObject, val: *mut NPY_ORDER) -> c_int;
    pub fn PyArray_EquivTypes(type1: *mut PyArray_Descr, type2: *mut PyArray_Descr) -> c_uchar;
    pub fn PyArray_Zeros(nd: c_int, dims: *mut npy_intp, type_: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject;
    pub fn PyArray_Empty(nd: c_int, dims: *mut npy_intp, type_: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject;
    pub fn PyArray_Where(condition: *mut PyObject, x: *mut PyObject, y: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_Arange(start: f64, stop: f64, step: f64, type_num: c_int) -> *mut PyObject;
    pub fn PyArray_ArangeObj(start: *mut PyObject, stop: *mut PyObject, step: *mut PyObject, dtype: *mut PyArray_Descr) -> *mut PyObject;
    pub fn PyArray_SortkindConverter(obj: *mut PyObject, sortkind: *mut NPY_SORTKIND) -> c_int;
    pub fn PyArray_LexSort(sort_keys: *mut PyObject, axis: c_int) -> *mut PyObject;
    pub fn PyArray_Round(a: *mut PyArrayObject, decimals: c_int, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_EquivTypenums(typenum1: c_int, typenum2: c_int) -> c_uchar;
    pub fn PyArray_RegisterDataType(descr: *mut PyArray_Descr) -> c_int;
    pub fn PyArray_RegisterCastFunc(descr: *mut PyArray_Descr, totype: c_int, castfunc: PyArray_VectorUnaryFunc) -> c_int;
    pub fn PyArray_RegisterCanCast(descr: *mut PyArray_Descr, totype: c_int, scalar: NPY_SCALARKIND) -> c_int;
    pub fn PyArray_InitArrFuncs(f: *mut PyArray_ArrFuncs);
    pub fn PyArray_IntTupleFromIntp(len: c_int, vals: *mut npy_intp) -> *mut PyObject;
    pub fn PyArray_TypeNumFromName(str: *mut c_char) -> c_int;
    pub fn PyArray_ClipmodeConverter(object: *mut PyObject, val: *mut NPY_CLIPMODE) -> c_int;
    pub fn PyArray_OutputConverter(object: *mut PyObject, address: *mut *mut PyArrayObject) -> c_int;
    pub fn PyArray_BroadcastToShape(obj: *mut PyObject, dims: *mut npy_intp, nd: c_int) -> *mut PyObject;
    pub fn PyArray_DescrAlignConverter(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int;
    pub fn PyArray_DescrAlignConverter2(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int;
    pub fn PyArray_SearchsideConverter(obj: *mut PyObject, addr: *mut c_void) -> c_int;
    pub fn PyArray_CheckAxis(arr: *mut PyArrayObject, axis: *mut c_int, flags: c_int) -> *mut PyObject;
    pub fn PyArray_OverflowMultiplyList(l1: *mut npy_intp, n: c_int) -> npy_intp;
    pub fn PyArray_CompareString(s1: *mut c_char, s2: *mut c_char, len: usize) -> c_int;
    pub fn PyArray_MultiIterFromObjects(mps: *mut *mut PyObject, n: c_int, nadd: c_int, ...) -> *mut PyObject;
    pub fn PyArray_GetEndianness() -> c_int;
    pub fn PyArray_GetNDArrayCFeatureVersion() -> c_uint;
    pub fn PyArray_Correlate2(op1: *mut PyObject, op2: *mut PyObject, mode: c_int) -> *mut PyObject;
    pub fn PyArray_NeighborhoodIterNew(x: *mut PyArrayIterObject, bounds: *mut npy_intp, mode: c_int, fill: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_SetDatetimeParseFunction(op: *mut PyObject);
    pub fn PyArray_DatetimeToDatetimeStruct(val: npy_datetime, fr: NPY_DATETIMEUNIT, result: *mut npy_datetimestruct);
    pub fn PyArray_TimedeltaToTimedeltaStruct(val: npy_timedelta, fr: NPY_DATETIMEUNIT, result: *mut npy_timedeltastruct);
    pub fn PyArray_DatetimeStructToDatetime(fr: NPY_DATETIMEUNIT, d: *mut npy_datetimestruct) -> npy_datetime;
    pub fn PyArray_TimedeltaStructToTimedelta(fr: NPY_DATETIMEUNIT, d: *mut npy_timedeltastruct) -> npy_datetime;
    pub fn PyArray_CastingConverter(obj: *mut PyObject, casting: *mut NPY_CASTING) -> c_int;
    pub fn PyArray_CountNonzero(self_: *mut PyArrayObject) -> npy_intp;
    pub fn PyArray_PromoteTypes(type1: *mut PyArray_Descr, type2: *mut PyArray_Descr) -> *mut PyArray_Descr;
    pub fn PyArray_MinScalarType(arr: *mut PyArrayObject) -> *mut PyArray_Descr;
    pub fn PyArray_ResultType(narrs: npy_intp, arr: *mut *mut PyArrayObject, ndtypes: npy_intp, dtypes: *mut *mut PyArray_Descr) -> *mut PyArray_Descr;
    pub fn PyArray_CanCastArrayTo(arr: *mut PyArrayObject, to: *mut PyArray_Descr, casting: NPY_CASTING) -> npy_bool;
    pub fn PyArray_CanCastTypeTo(from: *mut PyArray_Descr, to: *mut PyArray_Descr, casting: NPY_CASTING) -> npy_bool;
    pub fn PyArray_EinsteinSum(subscripts: *mut c_char,
                               nop: npy_intp,
                               op_in: *mut *mut PyArrayObject,
                               dtype: *mut PyArray_Descr,
                               order: NPY_ORDER,
                               casting: NPY_CASTING,
                               out: *mut PyArrayObject)
                               -> *mut PyArrayObject;
    pub fn PyArray_NewLikeArray(prototype: *mut PyArrayObject, order: NPY_ORDER, dtype: *mut PyArray_Descr, subok: c_int) -> *mut PyObject;
    pub fn PyArray_GetArrayParamsFromObject(op: *mut PyObject,
                                            requested_dtype: *mut PyArray_Descr,
                                            writeable: npy_bool,
                                            out_dtype: *mut *mut PyArray_Descr,
                                            out_ndim: *mut c_int,
                                            out_dims: *mut npy_intp,
                                            out_arr: *mut *mut PyArrayObject,
                                            context: *mut PyObject)
                                            -> c_int;
    pub fn PyArray_ConvertClipmodeSequence(object: *mut PyObject, modes: *mut NPY_CLIPMODE, n: c_int) -> c_int;
    pub fn PyArray_MatrixProduct2(op1: *mut PyObject, op2: *mut PyObject, out: *mut PyArrayObject) -> *mut PyObject;
    pub fn PyArray_SetBaseObject(arr: *mut PyArrayObject, obj: *mut PyObject) -> c_int;
    pub fn PyArray_CreateSortedStridePerm(ndim: c_int, strides: *mut npy_intp, out_strideperm: *mut npy_stride_sort_item);
    pub fn PyArray_RemoveAxesInPlace(arr: *mut PyArrayObject, flags: *mut npy_bool);
    pub fn PyArray_DebugPrint(obj: *mut PyArrayObject);
    pub fn PyArray_FailUnlessWriteable(obj: *mut PyArrayObject, name: *const c_char) -> c_int;
    pub fn PyArray_SetUpdateIfCopyBase(arr: *mut PyArrayObject, base: *mut PyArrayObject) -> c_int;
    pub fn PyArray_MapIterSwapAxes(mit: *mut PyArrayMapIterObject, ret: *mut *mut PyArrayObject, getmap: c_int);
    pub fn PyArray_MapIterArray(a: *mut PyArrayObject, index: *mut PyObject) -> *mut PyObject;
    pub fn PyArray_MapIterNext(mit: *mut PyArrayMapIterObject);
    pub fn PyArray_Partition(op: *mut PyArrayObject, ktharray: *mut PyArrayObject, axis: c_int, which: NPY_SELECTKIND) -> c_int;
    pub fn PyArray_ArgPartition(op: *mut PyArrayObject, ktharray: *mut PyArrayObject, axis: c_int, which: NPY_SELECTKIND) -> *mut PyObject;
    pub fn PyArray_SelectkindConverter(obj: *mut PyObject, selectkind: *mut NPY_SELECTKIND) -> c_int;
    pub fn PyArray_CheckAnyScalarExact(obj: *mut PyObject) -> c_int;
}

extern "C" {
    pub fn PyDataMem_NEW(size: usize) -> *mut c_void;
    pub fn PyDataMem_FREE(ptr: *mut c_void);
    pub fn PyDataMem_RENEW(ptr: *mut c_void, size: usize) -> *mut c_void;
    pub fn PyDataMem_SetEventHook(newhook: PyDataMem_EventHookFunc, user_data: *mut c_void, old_data: *mut *mut c_void) -> PyDataMem_EventHookFunc;
    pub fn PyDataMem_NEW_ZEROED(size: usize, elsize: usize) -> *mut c_void;
}

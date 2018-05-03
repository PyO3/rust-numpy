//! Low-Level binding for Array API
//!
//! https://docs.scipy.org/doc/numpy/reference/c-api.array.html

use libc::FILE;
use std::ops::Deref;
use std::os::raw::*;
use std::ptr::null_mut;

use pyo3::ffi;
use pyo3::ffi::{PyObject, PyTypeObject};
use pyo3::{ObjectProtocol, PyModule, PyResult, Python, ToPyPointer};

use npyffi::*;

/// Low-Level binding for Array API
/// https://docs.scipy.org/doc/numpy/reference/c-api.array.html
///
/// Most of Array API is exposed as the related function of this module object.
/// Some APIs (including most accessor) in the above URL are not implemented
/// since they are defined as C macro, and cannot be called from rust.
/// Some of these are implemented on the high-level interface as a Rust function.
pub struct PyArrayModule<'py> {
    numpy: &'py PyModule,
    api: *const *const c_void,
}

impl<'py> Deref for PyArrayModule<'py> {
    type Target = PyModule;
    fn deref(&self) -> &Self::Target {
        self.numpy
    }
}

/// Define Array API in PyArrayModule
macro_rules! pyarray_api {
    [ $offset:expr; $fname:ident ( $($arg:ident : $t:ty),* ) $( -> $ret:ty )* ] => {
#[allow(non_snake_case)]
pub unsafe fn $fname(&self, $($arg : $t), *) $( -> $ret )* {
    let fptr = self.api.offset($offset) as (*const extern fn ($($arg : $t), *) $( -> $ret )* );
    (*fptr)($($arg), *)
}
}} // pyarray_api!

impl<'py> PyArrayModule<'py> {
    /// Import `numpy.core.multiarray` to use Array API.
    pub fn import(py: Python<'py>) -> PyResult<Self> {
        let numpy = py.import("numpy.core.multiarray")?;
        let c_api = numpy.getattr("_ARRAY_API")?;
        let api = unsafe {
            ffi::PyCapsule_GetPointer(c_api.as_ptr(), null_mut()) as *const *const c_void
        };
        Ok(Self {
            numpy: numpy,
            api: api,
        })
    }

    pyarray_api![0; PyArray_GetNDArrayCVersion() -> c_uint];
    pyarray_api![40; PyArray_SetNumericOps(dict: *mut PyObject) -> c_int];
    pyarray_api![41; PyArray_GetNumericOps() -> *mut PyObject];
    pyarray_api![42; PyArray_INCREF(mp: *mut PyArrayObject) -> c_int];
    pyarray_api![43; PyArray_XDECREF(mp: *mut PyArrayObject) -> c_int];
    pyarray_api![44; PyArray_SetStringFunction(op: *mut PyObject, repr: c_int)];
    pyarray_api![45; PyArray_DescrFromType(type_: c_int) -> *mut PyArray_Descr];
    pyarray_api![46; PyArray_TypeObjectFromType(type_: c_int) -> *mut PyObject];
    pyarray_api![47; PyArray_Zero(arr: *mut PyArrayObject) -> *mut c_char];
    pyarray_api![48; PyArray_One(arr: *mut PyArrayObject) -> *mut c_char];
    pyarray_api![49; PyArray_CastToType(arr: *mut PyArrayObject, dtype: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject];
    pyarray_api![50; PyArray_CastTo(out: *mut PyArrayObject, mp: *mut PyArrayObject) -> c_int];
    pyarray_api![51; PyArray_CastAnyTo(out: *mut PyArrayObject, mp: *mut PyArrayObject) -> c_int];
    pyarray_api![52; PyArray_CanCastSafely(fromtype: c_int, totype: c_int) -> c_int];
    pyarray_api![53; PyArray_CanCastTo(from: *mut PyArray_Descr, to: *mut PyArray_Descr) -> npy_bool];
    pyarray_api![54; PyArray_ObjectType(op: *mut PyObject, minimum_type: c_int) -> c_int];
    pyarray_api![55; PyArray_DescrFromObject(op: *mut PyObject, mintype: *mut PyArray_Descr) -> *mut PyArray_Descr];
    pyarray_api![56; PyArray_ConvertToCommonType(op: *mut PyObject, retn: *mut c_int) -> *mut *mut PyArrayObject];
    pyarray_api![57; PyArray_DescrFromScalar(sc: *mut PyObject) -> *mut PyArray_Descr];
    pyarray_api![58; PyArray_DescrFromTypeObject(type_: *mut PyObject) -> *mut PyArray_Descr];
    pyarray_api![59; PyArray_Size(op: *mut PyObject) -> npy_intp];
    pyarray_api![60; PyArray_Scalar(data: *mut c_void, descr: *mut PyArray_Descr, base: *mut PyObject) -> *mut PyObject];
    pyarray_api![61; PyArray_FromScalar(scalar: *mut PyObject, outcode: *mut PyArray_Descr) -> *mut PyObject];
    pyarray_api![62; PyArray_ScalarAsCtype(scalar: *mut PyObject, ctypeptr: *mut c_void)];
    pyarray_api![63; PyArray_CastScalarToCtype(scalar: *mut PyObject, ctypeptr: *mut c_void, outcode: *mut PyArray_Descr) -> c_int];
    pyarray_api![64; PyArray_CastScalarDirect(scalar: *mut PyObject, indescr: *mut PyArray_Descr, ctypeptr: *mut c_void, outtype: c_int) -> c_int];
    pyarray_api![65; PyArray_ScalarFromObject(object: *mut PyObject) -> *mut PyObject];
    pyarray_api![66; PyArray_GetCastFunc(descr: *mut PyArray_Descr, type_num: c_int) -> PyArray_VectorUnaryFunc];
    pyarray_api![67; PyArray_FromDims(nd: c_int, d: *mut c_int, type_: c_int) -> *mut PyObject];
    pyarray_api![68; PyArray_FromDimsAndDataAndDescr(nd: c_int, d: *mut c_int, descr: *mut PyArray_Descr, data: *mut c_char) -> *mut PyObject];
    pyarray_api![69; PyArray_FromAny(op: *mut PyObject, newtype: *mut PyArray_Descr, min_depth: c_int, max_depth: c_int, flags: c_int, context: *mut PyObject) -> *mut PyObject];
    pyarray_api![70; PyArray_EnsureArray(op: *mut PyObject) -> *mut PyObject];
    pyarray_api![71; PyArray_EnsureAnyArray(op: *mut PyObject) -> *mut PyObject];
    pyarray_api![72; PyArray_FromFile(fp: *mut FILE, dtype: *mut PyArray_Descr, num: npy_intp, sep: *mut c_char) -> *mut PyObject];
    pyarray_api![73; PyArray_FromString(data: *mut c_char, slen: npy_intp, dtype: *mut PyArray_Descr, num: npy_intp, sep: *mut c_char) -> *mut PyObject];
    pyarray_api![74; PyArray_FromBuffer(buf: *mut PyObject, type_: *mut PyArray_Descr, count: npy_intp, offset: npy_intp) -> *mut PyObject];
    pyarray_api![75; PyArray_FromIter(obj: *mut PyObject, dtype: *mut PyArray_Descr, count: npy_intp) -> *mut PyObject];
    pyarray_api![76; PyArray_Return(mp: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![77; PyArray_GetField(self_: *mut PyArrayObject, typed: *mut PyArray_Descr, offset: c_int) -> *mut PyObject];
    pyarray_api![78; PyArray_SetField(self_: *mut PyArrayObject, dtype: *mut PyArray_Descr, offset: c_int, val: *mut PyObject) -> c_int];
    pyarray_api![79; PyArray_Byteswap(self_: *mut PyArrayObject, inplace: npy_bool) -> *mut PyObject];
    pyarray_api![80; PyArray_Resize(self_: *mut PyArrayObject, newshape: *mut PyArray_Dims, refcheck: c_int, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![81; PyArray_MoveInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int];
    pyarray_api![82; PyArray_CopyInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int];
    pyarray_api![83; PyArray_CopyAnyInto(dst: *mut PyArrayObject, src: *mut PyArrayObject) -> c_int];
    pyarray_api![84; PyArray_CopyObject(dest: *mut PyArrayObject, src_object: *mut PyObject) -> c_int];
    pyarray_api![85; PyArray_NewCopy(obj: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![86; PyArray_ToList(self_: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![87; PyArray_ToString(self_: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![88; PyArray_ToFile(self_: *mut PyArrayObject, fp: *mut FILE, sep: *mut c_char, format: *mut c_char) -> c_int];
    pyarray_api![89; PyArray_Dump(self_: *mut PyObject, file: *mut PyObject, protocol: c_int) -> c_int];
    pyarray_api![90; PyArray_Dumps(self_: *mut PyObject, protocol: c_int) -> *mut PyObject];
    pyarray_api![91; PyArray_ValidType(type_: c_int) -> c_int];
    pyarray_api![92; PyArray_UpdateFlags(ret: *mut PyArrayObject, flagmask: c_int)];
    pyarray_api![93; PyArray_New(subtype: *mut PyTypeObject, nd: c_int, dims: *mut npy_intp, type_num: c_int, strides: *mut npy_intp, data: *mut c_void, itemsize: c_int, flags: c_int, obj: *mut PyObject) -> *mut PyObject];
    pyarray_api![94; PyArray_NewFromDescr(subtype: *mut PyTypeObject, descr: *mut PyArray_Descr, nd: c_int, dims: *mut npy_intp, strides: *mut npy_intp, data: *mut c_void, flags: c_int, obj: *mut PyObject) -> *mut PyObject];
    pyarray_api![95; PyArray_DescrNew(base: *mut PyArray_Descr) -> *mut PyArray_Descr];
    pyarray_api![96; PyArray_DescrNewFromType(type_num: c_int) -> *mut PyArray_Descr];
    pyarray_api![97; PyArray_GetPriority(obj: *mut PyObject, default_: f64) -> f64];
    pyarray_api![98; PyArray_IterNew(obj: *mut PyObject) -> *mut PyObject];
    // pyarray_api![99; PyArray_MultiIterNew(n: c_int, ...) -> *mut PyObject];
    pyarray_api![100; PyArray_PyIntAsInt(o: *mut PyObject) -> c_int];
    pyarray_api![101; PyArray_PyIntAsIntp(o: *mut PyObject) -> npy_intp];
    pyarray_api![102; PyArray_Broadcast(mit: *mut PyArrayMultiIterObject) -> c_int];
    pyarray_api![103; PyArray_FillObjectArray(arr: *mut PyArrayObject, obj: *mut PyObject)];
    pyarray_api![104; PyArray_FillWithScalar(arr: *mut PyArrayObject, obj: *mut PyObject) -> c_int];
    pyarray_api![105; PyArray_CheckStrides(elsize: c_int, nd: c_int, numbytes: npy_intp, offset: npy_intp, dims: *mut npy_intp, newstrides: *mut npy_intp) -> npy_bool];
    pyarray_api![106; PyArray_DescrNewByteorder(self_: *mut PyArray_Descr, newendian: c_char) -> *mut PyArray_Descr];
    pyarray_api![107; PyArray_IterAllButAxis(obj: *mut PyObject, inaxis: *mut c_int) -> *mut PyObject];
    pyarray_api![108; PyArray_CheckFromAny(op: *mut PyObject, descr: *mut PyArray_Descr, min_depth: c_int, max_depth: c_int, requires: c_int, context: *mut PyObject) -> *mut PyObject];
    pyarray_api![109; PyArray_FromArray(arr: *mut PyArrayObject, newtype: *mut PyArray_Descr, flags: c_int) -> *mut PyObject];
    pyarray_api![110; PyArray_FromInterface(origin: *mut PyObject) -> *mut PyObject];
    pyarray_api![111; PyArray_FromStructInterface(input: *mut PyObject) -> *mut PyObject];
    pyarray_api![112; PyArray_FromArrayAttr(op: *mut PyObject, typecode: *mut PyArray_Descr, context: *mut PyObject) -> *mut PyObject];
    pyarray_api![113; PyArray_ScalarKind(typenum: c_int, arr: *mut *mut PyArrayObject) -> NPY_SCALARKIND];
    pyarray_api![114; PyArray_CanCoerceScalar(thistype: c_int, neededtype: c_int, scalar: NPY_SCALARKIND) -> c_int];
    pyarray_api![115; PyArray_NewFlagsObject(obj: *mut PyObject) -> *mut PyObject];
    pyarray_api![116; PyArray_CanCastScalar(from: *mut PyTypeObject, to: *mut PyTypeObject) -> npy_bool];
    pyarray_api![117; PyArray_CompareUCS4(s1: *mut npy_ucs4, s2: *mut npy_ucs4, len: usize) -> c_int];
    pyarray_api![118; PyArray_RemoveSmallest(multi: *mut PyArrayMultiIterObject) -> c_int];
    pyarray_api![119; PyArray_ElementStrides(obj: *mut PyObject) -> c_int];
    pyarray_api![120; PyArray_Item_INCREF(data: *mut c_char, descr: *mut PyArray_Descr)];
    pyarray_api![121; PyArray_Item_XDECREF(data: *mut c_char, descr: *mut PyArray_Descr)];
    pyarray_api![122; PyArray_FieldNames(fields: *mut PyObject) -> *mut PyObject];
    pyarray_api![123; PyArray_Transpose(ap: *mut PyArrayObject, permute: *mut PyArray_Dims) -> *mut PyObject];
    pyarray_api![124; PyArray_TakeFrom(self0: *mut PyArrayObject, indices0: *mut PyObject, axis: c_int, out: *mut PyArrayObject, clipmode: NPY_CLIPMODE) -> *mut PyObject];
    pyarray_api![125; PyArray_PutTo(self_: *mut PyArrayObject, values0: *mut PyObject, indices0: *mut PyObject, clipmode: NPY_CLIPMODE) -> *mut PyObject];
    pyarray_api![126; PyArray_PutMask(self_: *mut PyArrayObject, values0: *mut PyObject, mask0: *mut PyObject) -> *mut PyObject];
    pyarray_api![127; PyArray_Repeat(aop: *mut PyArrayObject, op: *mut PyObject, axis: c_int) -> *mut PyObject];
    pyarray_api![128; PyArray_Choose(ip: *mut PyArrayObject, op: *mut PyObject, out: *mut PyArrayObject, clipmode: NPY_CLIPMODE) -> *mut PyObject];
    pyarray_api![129; PyArray_Sort(op: *mut PyArrayObject, axis: c_int, which: NPY_SORTKIND) -> c_int];
    pyarray_api![130; PyArray_ArgSort(op: *mut PyArrayObject, axis: c_int, which: NPY_SORTKIND) -> *mut PyObject];
    pyarray_api![131; PyArray_SearchSorted(op1: *mut PyArrayObject, op2: *mut PyObject, side: NPY_SEARCHSIDE, perm: *mut PyObject) -> *mut PyObject];
    pyarray_api![132; PyArray_ArgMax(op: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![133; PyArray_ArgMin(op: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![134; PyArray_Reshape(self_: *mut PyArrayObject, shape: *mut PyObject) -> *mut PyObject];
    pyarray_api![135; PyArray_Newshape(self_: *mut PyArrayObject, newdims: *mut PyArray_Dims, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![136; PyArray_Squeeze(self_: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![137; PyArray_View(self_: *mut PyArrayObject, type_: *mut PyArray_Descr, pytype: *mut PyTypeObject) -> *mut PyObject];
    pyarray_api![138; PyArray_SwapAxes(ap: *mut PyArrayObject, a1: c_int, a2: c_int) -> *mut PyObject];
    pyarray_api![139; PyArray_Max(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![140; PyArray_Min(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![141; PyArray_Ptp(ap: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![142; PyArray_Mean(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![143; PyArray_Trace(self_: *mut PyArrayObject, offset: c_int, axis1: c_int, axis2: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![144; PyArray_Diagonal(self_: *mut PyArrayObject, offset: c_int, axis1: c_int, axis2: c_int) -> *mut PyObject];
    pyarray_api![145; PyArray_Clip(self_: *mut PyArrayObject, min: *mut PyObject, max: *mut PyObject, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![146; PyArray_Conjugate(self_: *mut PyArrayObject, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![147; PyArray_Nonzero(self_: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![148; PyArray_Std(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject, variance: c_int) -> *mut PyObject];
    pyarray_api![149; PyArray_Sum(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![150; PyArray_CumSum(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![151; PyArray_Prod(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![152; PyArray_CumProd(self_: *mut PyArrayObject, axis: c_int, rtype: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![153; PyArray_All(self_: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![154; PyArray_Any(self_: *mut PyArrayObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![155; PyArray_Compress(self_: *mut PyArrayObject, condition: *mut PyObject, axis: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![156; PyArray_Flatten(a: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![157; PyArray_Ravel(arr: *mut PyArrayObject, order: NPY_ORDER) -> *mut PyObject];
    pyarray_api![158; PyArray_MultiplyList(l1: *mut npy_intp, n: c_int) -> npy_intp];
    pyarray_api![159; PyArray_MultiplyIntList(l1: *mut c_int, n: c_int) -> c_int];
    pyarray_api![160; PyArray_GetPtr(obj: *mut PyArrayObject, ind: *mut npy_intp) -> *mut c_void];
    pyarray_api![161; PyArray_CompareLists(l1: *mut npy_intp, l2: *mut npy_intp, n: c_int) -> c_int];
    pyarray_api![162; PyArray_AsCArray(op: *mut *mut PyObject, ptr: *mut c_void, dims: *mut npy_intp, nd: c_int, typedescr: *mut PyArray_Descr) -> c_int];
    pyarray_api![163; PyArray_As1D(op: *mut *mut PyObject, ptr: *mut *mut c_char, d1: *mut c_int, typecode: c_int) -> c_int];
    pyarray_api![164; PyArray_As2D(op: *mut *mut PyObject, ptr: *mut *mut *mut c_char, d1: *mut c_int, d2: *mut c_int, typecode: c_int) -> c_int];
    pyarray_api![165; PyArray_Free(op: *mut PyObject, ptr: *mut c_void) -> c_int];
    pyarray_api![166; PyArray_Converter(object: *mut PyObject, address: *mut *mut PyObject) -> c_int];
    pyarray_api![167; PyArray_IntpFromSequence(seq: *mut PyObject, vals: *mut npy_intp, maxvals: c_int) -> c_int];
    pyarray_api![168; PyArray_Concatenate(op: *mut PyObject, axis: c_int) -> *mut PyObject];
    pyarray_api![169; PyArray_InnerProduct(op1: *mut PyObject, op2: *mut PyObject) -> *mut PyObject];
    pyarray_api![170; PyArray_MatrixProduct(op1: *mut PyObject, op2: *mut PyObject) -> *mut PyObject];
    pyarray_api![171; PyArray_CopyAndTranspose(op: *mut PyObject) -> *mut PyObject];
    pyarray_api![172; PyArray_Correlate(op1: *mut PyObject, op2: *mut PyObject, mode: c_int) -> *mut PyObject];
    pyarray_api![173; PyArray_TypestrConvert(itemsize: c_int, gentype: c_int) -> c_int];
    pyarray_api![174; PyArray_DescrConverter(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int];
    pyarray_api![175; PyArray_DescrConverter2(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int];
    pyarray_api![176; PyArray_IntpConverter(obj: *mut PyObject, seq: *mut PyArray_Dims) -> c_int];
    pyarray_api![177; PyArray_BufferConverter(obj: *mut PyObject, buf: *mut PyArray_Chunk) -> c_int];
    pyarray_api![178; PyArray_AxisConverter(obj: *mut PyObject, axis: *mut c_int) -> c_int];
    pyarray_api![179; PyArray_BoolConverter(object: *mut PyObject, val: *mut npy_bool) -> c_int];
    pyarray_api![180; PyArray_ByteorderConverter(obj: *mut PyObject, endian: *mut c_char) -> c_int];
    pyarray_api![181; PyArray_OrderConverter(object: *mut PyObject, val: *mut NPY_ORDER) -> c_int];
    pyarray_api![182; PyArray_EquivTypes(type1: *mut PyArray_Descr, type2: *mut PyArray_Descr) -> c_uchar];
    pyarray_api![183; PyArray_Zeros(nd: c_int, dims: *mut npy_intp, type_: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject];
    pyarray_api![184; PyArray_Empty(nd: c_int, dims: *mut npy_intp, type_: *mut PyArray_Descr, is_f_order: c_int) -> *mut PyObject];
    pyarray_api![185; PyArray_Where(condition: *mut PyObject, x: *mut PyObject, y: *mut PyObject) -> *mut PyObject];
    pyarray_api![186; PyArray_Arange(start: f64, stop: f64, step: f64, type_num: c_int) -> *mut PyObject];
    pyarray_api![187; PyArray_ArangeObj(start: *mut PyObject, stop: *mut PyObject, step: *mut PyObject, dtype: *mut PyArray_Descr) -> *mut PyObject];
    pyarray_api![188; PyArray_SortkindConverter(obj: *mut PyObject, sortkind: *mut NPY_SORTKIND) -> c_int];
    pyarray_api![189; PyArray_LexSort(sort_keys: *mut PyObject, axis: c_int) -> *mut PyObject];
    pyarray_api![190; PyArray_Round(a: *mut PyArrayObject, decimals: c_int, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![191; PyArray_EquivTypenums(typenum1: c_int, typenum2: c_int) -> c_uchar];
    pyarray_api![192; PyArray_RegisterDataType(descr: *mut PyArray_Descr) -> c_int];
    pyarray_api![193; PyArray_RegisterCastFunc(descr: *mut PyArray_Descr, totype: c_int, castfunc: PyArray_VectorUnaryFunc) -> c_int];
    pyarray_api![194; PyArray_RegisterCanCast(descr: *mut PyArray_Descr, totype: c_int, scalar: NPY_SCALARKIND) -> c_int];
    pyarray_api![195; PyArray_InitArrFuncs(f: *mut PyArray_ArrFuncs)];
    pyarray_api![196; PyArray_IntTupleFromIntp(len: c_int, vals: *mut npy_intp) -> *mut PyObject];
    pyarray_api![197; PyArray_TypeNumFromName(str: *mut c_char) -> c_int];
    pyarray_api![198; PyArray_ClipmodeConverter(object: *mut PyObject, val: *mut NPY_CLIPMODE) -> c_int];
    pyarray_api![199; PyArray_OutputConverter(object: *mut PyObject, address: *mut *mut PyArrayObject) -> c_int];
    pyarray_api![200; PyArray_BroadcastToShape(obj: *mut PyObject, dims: *mut npy_intp, nd: c_int) -> *mut PyObject];
    pyarray_api![201; _PyArray_SigintHandler(signum: c_int)];
    pyarray_api![202; _PyArray_GetSigintBuf() -> *mut c_void];
    pyarray_api![203; PyArray_DescrAlignConverter(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int];
    pyarray_api![204; PyArray_DescrAlignConverter2(obj: *mut PyObject, at: *mut *mut PyArray_Descr) -> c_int];
    pyarray_api![205; PyArray_SearchsideConverter(obj: *mut PyObject, addr: *mut c_void) -> c_int];
    pyarray_api![206; PyArray_CheckAxis(arr: *mut PyArrayObject, axis: *mut c_int, flags: c_int) -> *mut PyObject];
    pyarray_api![207; PyArray_OverflowMultiplyList(l1: *mut npy_intp, n: c_int) -> npy_intp];
    pyarray_api![208; PyArray_CompareString(s1: *mut c_char, s2: *mut c_char, len: usize) -> c_int];
    // pyarray_api![209; PyArray_MultiIterFromObjects(mps: *mut *mut PyObject, n: c_int, nadd: c_int, ...) -> *mut PyObject];
    pyarray_api![210; PyArray_GetEndianness() -> c_int];
    pyarray_api![211; PyArray_GetNDArrayCFeatureVersion() -> c_uint];
    pyarray_api![212; PyArray_Correlate2(op1: *mut PyObject, op2: *mut PyObject, mode: c_int) -> *mut PyObject];
    pyarray_api![213; PyArray_NeighborhoodIterNew(x: *mut PyArrayIterObject, bounds: *mut npy_intp, mode: c_int, fill: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![219; PyArray_SetDatetimeParseFunction(op: *mut PyObject)];
    pyarray_api![220; PyArray_DatetimeToDatetimeStruct(val: npy_datetime, fr: NPY_DATETIMEUNIT, result: *mut npy_datetimestruct)];
    pyarray_api![221; PyArray_TimedeltaToTimedeltaStruct(val: npy_timedelta, fr: NPY_DATETIMEUNIT, result: *mut npy_timedeltastruct)];
    pyarray_api![222; PyArray_DatetimeStructToDatetime(fr: NPY_DATETIMEUNIT, d: *mut npy_datetimestruct) -> npy_datetime];
    pyarray_api![223; PyArray_TimedeltaStructToTimedelta(fr: NPY_DATETIMEUNIT, d: *mut npy_timedeltastruct) -> npy_datetime];
    pyarray_api![224; NpyIter_New(op: *mut PyArrayObject, flags: npy_uint32, order: NPY_ORDER, casting: NPY_CASTING, dtype: *mut PyArray_Descr) -> *mut NpyIter];
    pyarray_api![225; NpyIter_MultiNew(nop: c_int, op_in: *mut *mut PyArrayObject, flags: npy_uint32, order: NPY_ORDER, casting: NPY_CASTING, op_flags: *mut npy_uint32, op_request_dtypes: *mut *mut PyArray_Descr) -> *mut NpyIter];
    pyarray_api![226; NpyIter_AdvancedNew(nop: c_int, op_in: *mut *mut PyArrayObject, flags: npy_uint32, order: NPY_ORDER, casting: NPY_CASTING, op_flags: *mut npy_uint32, op_request_dtypes: *mut *mut PyArray_Descr, oa_ndim: c_int, op_axes: *mut *mut c_int, itershape: *mut npy_intp, buffersize: npy_intp) -> *mut NpyIter];
    pyarray_api![227; NpyIter_Copy(iter: *mut NpyIter) -> *mut NpyIter];
    pyarray_api![228; NpyIter_Deallocate(iter: *mut NpyIter) -> c_int];
    pyarray_api![229; NpyIter_HasDelayedBufAlloc(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![230; NpyIter_HasExternalLoop(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![231; NpyIter_EnableExternalLoop(iter: *mut NpyIter) -> c_int];
    pyarray_api![232; NpyIter_GetInnerStrideArray(iter: *mut NpyIter) -> *mut npy_intp];
    pyarray_api![233; NpyIter_GetInnerLoopSizePtr(iter: *mut NpyIter) -> *mut npy_intp];
    pyarray_api![234; NpyIter_Reset(iter: *mut NpyIter, errmsg: *mut *mut c_char) -> c_int];
    pyarray_api![235; NpyIter_ResetBasePointers(iter: *mut NpyIter, baseptrs: *mut *mut c_char, errmsg: *mut *mut c_char) -> c_int];
    pyarray_api![236; NpyIter_ResetToIterIndexRange(iter: *mut NpyIter, istart: npy_intp, iend: npy_intp, errmsg: *mut *mut c_char) -> c_int];
    pyarray_api![237; NpyIter_GetNDim(iter: *mut NpyIter) -> c_int];
    pyarray_api![238; NpyIter_GetNOp(iter: *mut NpyIter) -> c_int];
    pyarray_api![239; NpyIter_GetIterNext(iter: *mut NpyIter, errmsg: *mut *mut c_char) -> NpyIter_IterNextFunc];
    pyarray_api![240; NpyIter_GetIterSize(iter: *mut NpyIter) -> npy_intp];
    pyarray_api![241; NpyIter_GetIterIndexRange(iter: *mut NpyIter, istart: *mut npy_intp, iend: *mut npy_intp)];
    pyarray_api![242; NpyIter_GetIterIndex(iter: *mut NpyIter) -> npy_intp];
    pyarray_api![243; NpyIter_GotoIterIndex(iter: *mut NpyIter, iterindex: npy_intp) -> c_int];
    pyarray_api![244; NpyIter_HasMultiIndex(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![245; NpyIter_GetShape(iter: *mut NpyIter, outshape: *mut npy_intp) -> c_int];
    pyarray_api![246; NpyIter_GetGetMultiIndex(iter: *mut NpyIter, errmsg: *mut *mut c_char) -> NpyIter_GetMultiIndexFunc];
    pyarray_api![247; NpyIter_GotoMultiIndex(iter: *mut NpyIter, multi_index: *mut npy_intp) -> c_int];
    pyarray_api![248; NpyIter_RemoveMultiIndex(iter: *mut NpyIter) -> c_int];
    pyarray_api![249; NpyIter_HasIndex(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![250; NpyIter_IsBuffered(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![251; NpyIter_IsGrowInner(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![252; NpyIter_GetBufferSize(iter: *mut NpyIter) -> npy_intp];
    pyarray_api![253; NpyIter_GetIndexPtr(iter: *mut NpyIter) -> *mut npy_intp];
    pyarray_api![254; NpyIter_GotoIndex(iter: *mut NpyIter, flat_index: npy_intp) -> c_int];
    pyarray_api![255; NpyIter_GetDataPtrArray(iter: *mut NpyIter) -> *mut *mut c_char];
    pyarray_api![256; NpyIter_GetDescrArray(iter: *mut NpyIter) -> *mut *mut PyArray_Descr];
    pyarray_api![257; NpyIter_GetOperandArray(iter: *mut NpyIter) -> *mut *mut PyArrayObject];
    pyarray_api![258; NpyIter_GetIterView(iter: *mut NpyIter, i: npy_intp) -> *mut PyArrayObject];
    pyarray_api![259; NpyIter_GetReadFlags(iter: *mut NpyIter, outreadflags: *mut c_char)];
    pyarray_api![260; NpyIter_GetWriteFlags(iter: *mut NpyIter, outwriteflags: *mut c_char)];
    pyarray_api![261; NpyIter_DebugPrint(iter: *mut NpyIter)];
    pyarray_api![262; NpyIter_IterationNeedsAPI(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![263; NpyIter_GetInnerFixedStrideArray(iter: *mut NpyIter, out_strides: *mut npy_intp)];
    pyarray_api![264; NpyIter_RemoveAxis(iter: *mut NpyIter, axis: c_int) -> c_int];
    pyarray_api![265; NpyIter_GetAxisStrideArray(iter: *mut NpyIter, axis: c_int) -> *mut npy_intp];
    pyarray_api![266; NpyIter_RequiresBuffering(iter: *mut NpyIter) -> npy_bool];
    pyarray_api![267; NpyIter_GetInitialDataPtrArray(iter: *mut NpyIter) -> *mut *mut c_char];
    pyarray_api![268; NpyIter_CreateCompatibleStrides(iter: *mut NpyIter, itemsize: npy_intp, outstrides: *mut npy_intp) -> c_int];
    pyarray_api![269; PyArray_CastingConverter(obj: *mut PyObject, casting: *mut NPY_CASTING) -> c_int];
    pyarray_api![270; PyArray_CountNonzero(self_: *mut PyArrayObject) -> npy_intp];
    pyarray_api![271; PyArray_PromoteTypes(type1: *mut PyArray_Descr, type2: *mut PyArray_Descr) -> *mut PyArray_Descr];
    pyarray_api![272; PyArray_MinScalarType(arr: *mut PyArrayObject) -> *mut PyArray_Descr];
    pyarray_api![273; PyArray_ResultType(narrs: npy_intp, arr: *mut *mut PyArrayObject, ndtypes: npy_intp, dtypes: *mut *mut PyArray_Descr) -> *mut PyArray_Descr];
    pyarray_api![274; PyArray_CanCastArrayTo(arr: *mut PyArrayObject, to: *mut PyArray_Descr, casting: NPY_CASTING) -> npy_bool];
    pyarray_api![275; PyArray_CanCastTypeTo(from: *mut PyArray_Descr, to: *mut PyArray_Descr, casting: NPY_CASTING) -> npy_bool];
    pyarray_api![276; PyArray_EinsteinSum(subscripts: *mut c_char, nop: npy_intp, op_in: *mut *mut PyArrayObject, dtype: *mut PyArray_Descr, order: NPY_ORDER, casting: NPY_CASTING, out: *mut PyArrayObject) -> *mut PyArrayObject];
    pyarray_api![277; PyArray_NewLikeArray(prototype: *mut PyArrayObject, order: NPY_ORDER, dtype: *mut PyArray_Descr, subok: c_int) -> *mut PyObject];
    pyarray_api![278; PyArray_GetArrayParamsFromObject(op: *mut PyObject, requested_dtype: *mut PyArray_Descr, writeable: npy_bool, out_dtype: *mut *mut PyArray_Descr, out_ndim: *mut c_int, out_dims: *mut npy_intp, out_arr: *mut *mut PyArrayObject, context: *mut PyObject) -> c_int];
    pyarray_api![279; PyArray_ConvertClipmodeSequence(object: *mut PyObject, modes: *mut NPY_CLIPMODE, n: c_int) -> c_int];
    pyarray_api![280; PyArray_MatrixProduct2(op1: *mut PyObject, op2: *mut PyObject, out: *mut PyArrayObject) -> *mut PyObject];
    pyarray_api![281; NpyIter_IsFirstVisit(iter: *mut NpyIter, iop: c_int) -> npy_bool];
    pyarray_api![282; PyArray_SetBaseObject(arr: *mut PyArrayObject, obj: *mut PyObject) -> c_int];
    pyarray_api![283; PyArray_CreateSortedStridePerm(ndim: c_int, strides: *mut npy_intp, out_strideperm: *mut npy_stride_sort_item)];
    pyarray_api![284; PyArray_RemoveAxesInPlace(arr: *mut PyArrayObject, flags: *mut npy_bool)];
    pyarray_api![285; PyArray_DebugPrint(obj: *mut PyArrayObject)];
    pyarray_api![286; PyArray_FailUnlessWriteable(obj: *mut PyArrayObject, name: *const c_char) -> c_int];
    pyarray_api![287; PyArray_SetUpdateIfCopyBase(arr: *mut PyArrayObject, base: *mut PyArrayObject) -> c_int];
    pyarray_api![288; PyDataMem_NEW(size: usize) -> *mut c_void];
    pyarray_api![289; PyDataMem_FREE(ptr: *mut c_void)];
    pyarray_api![290; PyDataMem_RENEW(ptr: *mut c_void, size: usize) -> *mut c_void];
    pyarray_api![291; PyDataMem_SetEventHook(newhook: PyDataMem_EventHookFunc, user_data: *mut c_void, old_data: *mut *mut c_void) -> PyDataMem_EventHookFunc];
    pyarray_api![293; PyArray_MapIterSwapAxes(mit: *mut PyArrayMapIterObject, ret: *mut *mut PyArrayObject, getmap: c_int)];
    pyarray_api![294; PyArray_MapIterArray(a: *mut PyArrayObject, index: *mut PyObject) -> *mut PyObject];
    pyarray_api![295; PyArray_MapIterNext(mit: *mut PyArrayMapIterObject)];
    pyarray_api![296; PyArray_Partition(op: *mut PyArrayObject, ktharray: *mut PyArrayObject, axis: c_int, which: NPY_SELECTKIND) -> c_int];
    pyarray_api![297; PyArray_ArgPartition(op: *mut PyArrayObject, ktharray: *mut PyArrayObject, axis: c_int, which: NPY_SELECTKIND) -> *mut PyObject];
    pyarray_api![298; PyArray_SelectkindConverter(obj: *mut PyObject, selectkind: *mut NPY_SELECTKIND) -> c_int];
    pyarray_api![299; PyDataMem_NEW_ZEROED(size: usize, elsize: usize) -> *mut c_void];
    pyarray_api![300; PyArray_CheckAnyScalarExact(obj: *mut PyObject) -> c_int];
    pyarray_api![301; PyArray_MapIterArrayCopyIfOverlap(a: *mut PyArrayObject, index: *mut PyObject, copy_if_overlap: c_int, extra_op: *mut PyArrayObject) -> *mut PyObject];
} // impl PyArrayModule

/// Define PyTypeObject related to Array API
macro_rules! impl_array_type {
    ($(($offset:expr, $tname:ident)),*) => {
#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum ArrayType { $($tname),* }
impl<'py> PyArrayModule<'py> {
    pub unsafe fn get_type_object(&self, ty: ArrayType) -> *mut PyTypeObject {
        match ty {
            $( ArrayType::$tname => *(self.api.offset($offset)) as *mut PyTypeObject ),*
        }
    }
}
}} // impl_array_type!;

impl_array_type!(
    (1, PyBigArray_Type),
    (2, PyArray_Type),
    (3, PyArrayDescr_Type),
    (4, PyArrayFlags_Type),
    (5, PyArrayIter_Type),
    (6, PyArrayMultiIter_Type),
    (7, NPY_NUMUSERTYPES),
    (8, PyBoolArrType_Type),
    (9, _PyArrayScalar_BoolValues),
    (10, PyGenericArrType_Type),
    (11, PyNumberArrType_Type),
    (12, PyIntegerArrType_Type),
    (13, PySignedIntegerArrType_Type),
    (14, PyUnsignedIntegerArrType_Type),
    (15, PyInexactArrType_Type),
    (16, PyFloatingArrType_Type),
    (17, PyComplexFloatingArrType_Type),
    (18, PyFlexibleArrType_Type),
    (19, PyCharacterArrType_Type),
    (20, PyByteArrType_Type),
    (21, PyShortArrType_Type),
    (22, PyIntArrType_Type),
    (23, PyLongArrType_Type),
    (24, PyLongLongArrType_Type),
    (25, PyUByteArrType_Type),
    (26, PyUShortArrType_Type),
    (27, PyUIntArrType_Type),
    (28, PyULongArrType_Type),
    (29, PyULongLongArrType_Type),
    (30, PyFloatArrType_Type),
    (31, PyDoubleArrType_Type),
    (32, PyLongDoubleArrType_Type),
    (33, PyCFloatArrType_Type),
    (34, PyCDoubleArrType_Type),
    (35, PyCLongDoubleArrType_Type),
    (36, PyObjectArrType_Type),
    (37, PyStringArrType_Type),
    (38, PyUnicodeArrType_Type),
    (39, PyVoidArrType_Type)
);

#[allow(non_snake_case)]
pub unsafe fn PyArray_Check(np: &PyArrayModule, op: *mut PyObject) -> c_int {
    ffi::PyObject_TypeCheck(op, np.get_type_object(ArrayType::PyArray_Type))
}

#[allow(non_snake_case)]
pub unsafe fn PyArray_CheckExact(np: &PyArrayModule, op: *mut PyObject) -> c_int {
    (ffi::Py_TYPE(op) == np.get_type_object(ArrayType::PyArray_Type)) as c_int
}

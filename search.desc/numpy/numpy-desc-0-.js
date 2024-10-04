searchState.loadedDescShard("numpy", 0, "This crate provides Rust interfaces for NumPy C APIs, …\nMarker type to indicate that the element type received via …\nThe given array is already borrowed\nInidcates why borrowing an array failed.\nAlias for a <code>Complex&lt;f32&gt;</code>\nAlias for a <code>Complex&lt;f64&gt;</code>\nRepresents that a type can be an element of <code>PyArray</code>.\nRepresents that given <code>Vec</code> cannot be treated as an array.\nFlag that indicates whether this type is trivially …\nCreate a one-dimensional index\none-dimensional\nCreate a two-dimensional index\ntwo-dimensional\nCreate a three-dimensional index\nthree-dimensional\nCreate a four-dimensional index\nfour-dimensional\nCreate a five-dimensional index\nfive-dimensional\nCreate a six-dimensional index\nsix-dimensional\nCreate a dynamic-dimensional index\ndynamic-dimensional\nRepresents that the given array is not contiguous.\nThe given array is not writeable\nBinding of <code>numpy.dtype</code>.\nImplementation of functionality for <code>PyArrayDescr</code>.\nReceiver for arrays or array-like types.\nReceiver for zero-dimensional arrays or array-like types.\nReceiver for one-dimensional arrays or array-like types.\nReceiver for two-dimensional arrays or array-like types.\nReceiver for three-dimensional arrays or array-like types.\nReceiver for four-dimensional arrays or array-like types.\nReceiver for five-dimensional arrays or array-like types.\nReceiver for six-dimensional arrays or array-like types.\nReceiver for arrays or array-like types whose …\nA newtype wrapper around <code>[u8; N]</code> to handle <code>byte</code> scalars …\nA newtype wrapper around <code>[PyUCS4; N]</code> to handle <code>str_</code> scalars…\nA safe, untyped wrapper for NumPy’s <code>ndarray</code> class.\nImplementation of functionality for <code>PyUntypedArray</code>.\nMarker type to indicate that the element type received via …\nReturns the required alignment (bytes) of this type …\nReturns the required alignment (bytes) of this type …\nSafe interface for NumPy’s N-dimensional arrays\nCreate an <strong><code>Array</code></strong> with one, two or three dimensions.\nReturns a raw pointer to the underlying <code>PyArrayObject</code>.\nReturns a raw pointer to the underlying <code>PyArrayObject</code>.\nReturns <code>self</code> as <code>*mut PyArray_Descr</code>.\nReturns <code>self</code> as <code>*mut PyArray_Descr</code>.\nGets the underlying FFI pointer, returns a borrowed …\nGets the underlying FFI pointer, returns a borrowed …\nReturns the type descriptor for the base element of …\nReturns the type descriptor for the base element of …\nTypes to safely create references into NumPy arrays\nReturns an ASCII character indicating the byte-order of …\nReturns a unique ASCII character for each of the 21 …\nDefines conversion traits between Rust types and NumPy …\nSupport datetimes and timedeltas\nDeprecated form of <code>dot_bound</code>\nReturn the dot product of two arrays.\nReturns the type descriptor (“dtype”) for a registered …\nReturns the <code>dtype</code> of the array.\nReturns the <code>dtype</code> of the array.\nReturns the type descriptor (“dtype”) for a registered …\nDeprecated form of <code>einsum_bound</code>\nDeprecated form of <code>einsum_bound!</code>\nReturn the Einstein summation convention of given tensors.\nReturn the Einstein summation convention of given tensors.\nReturns bit-flags describing how this type descriptor is …\nReturns bit-flags describing how this type descriptor is …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the associated type descriptor (“dtype”) for …\nReturns the associated type descriptor (“dtype”) for …\nReturns the associated type descriptor (“dtype”) for …\nReturns the type descriptor and offset of the field with …\nReturns the type descriptor and offset of the field with …\nReturns true if the type descriptor is a structured type.\nReturns true if the type descriptor is a structured type.\nReturns true if the type descriptor contains any …\nReturns true if the type descriptor is a sub-array.\nReturns true if the type descriptor is a sub-array.\nImaginary portion of the complex number\nImaginary portion of the complex number\nDeprecated form of <code>inner_bound</code>\nReturn the inner product of two arrays.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns <code>self</code> as <code>*mut PyArray_Descr</code> while increasing the …\nReturns <code>self</code> as <code>*mut PyArray_Descr</code> while increasing the …\nReturns true if the type descriptor is a struct which …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the there are no elements in the array.\nReturns true if two type descriptors are equivalent.\nReturns true if two type descriptors are equivalent.\nReturns <code>true</code> if the internal data of the array is …\nReturns true if type descriptor byteorder is native, or …\nReturns the element size of this type descriptor.\nReturns the element size of this type descriptor.\nReturns an ASCII character (one of <code>biufcmMOSUV</code>) …\nCalculates the total number of elements in the array.\nReturns an ordered list of field names, or <code>None</code> if there …\nReturns an ordered list of field names, or <code>None</code> if there …\nReturns the number of dimensions if this type descriptor …\nReturns the number of dimensions if this type descriptor …\nReturns the number of dimensions of the array.\nCreates a new type descriptor (“dtype”) object from an …\nCreates a new type descriptor (“dtype”) object from an …\nLow-Level bindings for NumPy C API.\nReturns a unique number for each of the 21 different …\nShortcut for creating a type descriptor of <code>object</code> type.\nShortcut for creating a type descriptor of <code>object</code> type.\nReturns the type descriptor for a registered type.\nReturns the type descriptor for a registered type.\nA prelude\nDeprecated form of <code>pyarray_bound</code>\nCreate a <code>PyArray</code> with one, two or three dimensions.\nReal portion of the complex number\nReal portion of the complex number\nReturns the shape of the sub-array.\nReturns the shape of the sub-array.\nReturns a slice which contains dimmensions of the array.\nReturns a slice indicating how many bytes to advance when …\nReturns the array scalar corresponding to this type …\nReturns the array scalar corresponding to this type …\nA safe, statically-typed wrapper for NumPy’s <code>ndarray</code> …\nZero-dimensional array.\nImplementation of functionality for <code>PyArray0&lt;T&gt;</code>.\nOne-dimensional array.\nTwo-dimensional array.\nThree-dimensional array.\nFour-dimensional array.\nFive-dimensional array.\nSix-dimensional array.\nDynamic-dimensional array.\nImplementation of functionality for <code>PyArray&lt;T, D&gt;</code>.\nDeprecated form of <code>PyArray&lt;T, Ix1&gt;::arange_bound</code>\nReturn evenly spaced values within a given interval.\nReturns an <code>ArrayView</code> of the internal array.\nReturns an <code>ArrayView</code> of the internal array.\nReturns an <code>ArrayViewMut</code> of the internal array.\nReturns an <code>ArrayViewMut</code> of the internal array.\nReturns the internal array as <code>RawArrayView</code> enabling …\nReturns the internal array as <code>RawArrayView</code> enabling …\nReturns the internal array as <code>RawArrayViewMut</code> enabling …\nReturns the internal array as <code>RawArrayViewMut</code> enabling …\nReturns an immutable view of the internal data as a slice.\nReturns an immutable view of the internal data as a slice.\nReturns a mutable view of the internal data as a slice.\nReturns a mutable view of the internal data as a slice.\nAccess an untyped representation of this array.\nAccess an untyped representation of this array.\nDeprecated form of <code>PyArray&lt;T, D&gt;::borrow_from_array_bound</code>\nCreates a NumPy array backed by <code>array</code> and ties its …\nCast the <code>PyArray&lt;T&gt;</code> to <code>PyArray&lt;U&gt;</code>, by allocating a new …\nCast the <code>PyArray&lt;T&gt;</code> to <code>PyArray&lt;U&gt;</code>, by allocating a new …\nCopies <code>self</code> into <code>other</code>, performing a data type conversion …\nCopies <code>self</code> into <code>other</code>, performing a data type conversion …\nReturns a pointer to the first element of the array.\nReturns a pointer to the first element of the array.\nSame as <code>shape</code>, but returns <code>D</code> instead of <code>&amp;[usize]</code>.\nSame as <code>shape</code>, but returns <code>D</code> instead of <code>&amp;[usize]</code>.\nReturns the argument unchanged.\nDeprecated form of <code>PyArray&lt;T, D&gt;::from_array_bound</code>\nConstruct a NumPy array from a <code>ndarray::ArrayBase</code>.\nConstructs a reference to a <code>PyArray</code> from a raw point to a …\nDeprecated form of <code>PyArray&lt;T, Ix1&gt;::from_iter_bound</code>\nConstruct a one-dimensional array from an <code>Iterator</code>.\nDeprecated form of <code>PyArray&lt;T, D&gt;::from_owned_array_bound</code>\nConstructs a NumPy from an <code>ndarray::Array</code>\nDeprecated form of …\nConstruct a NumPy array containing objects stored in a …\nConstructs a reference to a <code>PyArray</code> from a raw pointer to …\nDeprecated form of <code>PyArray&lt;T, Ix1&gt;::from_slice_bound</code>\nConstruct a one-dimensional array from a slice.\nDeprecated form of <code>PyArray&lt;T, Ix1&gt;::from_vec_bound</code>\nDeprecated form of <code>PyArray&lt;T, Ix2&gt;::from_vec2_bound</code>\nConstruct a two-dimension array from a <code>Vec&lt;Vec&lt;T&gt;&gt;</code>.\nDeprecated form of <code>PyArray&lt;T, Ix3&gt;::from_vec3_bound</code>\nConstruct a three-dimensional array from a <code>Vec&lt;Vec&lt;Vec&lt;T&gt;&gt;&gt;</code>…\nConstruct a one-dimensional array from a <code>Vec&lt;T&gt;</code>.\nGet a reference of the specified element if the given …\nGet a reference of the specified element if the given …\nReturns a handle to NumPy’s multiarray module.\nSame as <code>get</code>, but returns <code>Option&lt;&amp;mut T&gt;</code>.\nSame as <code>get</code>, but returns <code>Option&lt;&amp;mut T&gt;</code>.\nGet a copy of the specified element in the array.\nGet a copy of the specified element in the array.\nCalls <code>U::from(self)</code>.\nGet the single element of a zero-dimensional array.\nGet the single element of a zero-dimensional array.\nDeprecated form of <code>PyArray&lt;T, D&gt;::new_bound</code>\nCreates a new uninitialized NumPy array.\nA view of <code>self</code> with a different order of axes determined …\nA view of <code>self</code> with a different order of axes determined …\nGet an immutable borrow of the NumPy array\nGet an immutable borrow of the NumPy array\nGet a mutable borrow of the NumPy array\nGet a mutable borrow of the NumPy array\nSpecial case of <code>reshape_with_order</code> which keeps the memory …\nSpecial case of <code>reshape_with_order</code> which keeps the memory …\nConstruct a new array which has same values as <code>self</code>, but …\nConstruct a new array which has same values as self, but …\nExtends or truncates the dimensions of an array.\nExtends or truncates the dimensions of an array.\nTurn an array with fixed dimensionality into one with …\nTurn an array with fixed dimensionality into one with …\nTurn <code>&amp;PyArray&lt;T,D&gt;</code> into <code>Py&lt;PyArray&lt;T,D&gt;&gt;</code>, i.e. a pointer …\nGet a copy of the array as an <code>ndarray::Array</code>.\nGet a copy of the array as an <code>ndarray::Array</code>.\nReturns a copy of the internal data of the array as a <code>Vec</code>.\nReturns a copy of the internal data of the array as a <code>Vec</code>.\nSpecial case of <code>permute</code> which reverses the order the axes.\nSpecial case of <code>permute</code> which reverses the order the axes.\nTry to convert this array into a <code>nalgebra::MatrixView</code> …\nTry to convert this array into a <code>nalgebra::MatrixView</code> …\nTry to convert this array into a <code>nalgebra::MatrixViewMut</code> …\nTry to convert this array into a <code>nalgebra::MatrixViewMut</code> …\nGet an immutable borrow of the NumPy array\nGet an immutable borrow of the NumPy array\nGet a mutable borrow of the NumPy array\nGet a mutable borrow of the NumPy array\nGet an immutable reference of the specified element, …\nGet an immutable reference of the specified element, …\nSame as <code>uget</code>, but returns <code>&amp;mut T</code>.\nSame as <code>uget</code>, but returns <code>&amp;mut T</code>.\nSame as <code>uget</code>, but returns <code>*mut T</code>.\nSame as <code>uget</code>, but returns <code>*mut T</code>.\nDeprecated form of <code>PyArray&lt;T, D&gt;::zeros_bound</code>\nConstruct a new NumPy array filled with zeros.\nRead-only borrow of an array.\nRead-only borrow of a zero-dimensional array.\nRead-only borrow of a one-dimensional array.\nRead-only borrow of a two-dimensional array.\nRead-only borrow of a three-dimensional array.\nRead-only borrow of a four-dimensional array.\nRead-only borrow of a five-dimensional array.\nRead-only borrow of a six-dimensional array.\nRead-only borrow of an array whose dimensionality is …\nRead-write borrow of an array.\nRead-write borrow of a zero-dimensional array.\nRead-write borrow of a one-dimensional array.\nRead-write borrow of a two-dimensional array.\nRead-write borrow of a three-dimensional array.\nRead-write borrow of a four-dimensional array.\nRead-write borrow of a five-dimensional array.\nRead-write borrow of a six-dimensional array.\nRead-write borrow of an array whose dimensionality is …\nProvides an immutable array view of the interior of the …\nProvides a mutable array view of the interior of the NumPy …\nConvert this two-dimensional array into a …\nConvert this one-dimensional array into a …\nConvert this one-dimensional array into a …\nConvert this two-dimensional array into a …\nProvide an immutable slice view of the interior of the …\nProvide a mutable slice view of the interior of the NumPy …\nReturns the argument unchanged.\nReturns the argument unchanged.\nProvide an immutable reference to an element of the NumPy …\nProvide a mutable reference to an element of the NumPy …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nExtends or truncates the dimensions of an array.\nTry to convert this array into a <code>nalgebra::MatrixView</code> …\nTry to convert this array into a <code>nalgebra::MatrixViewMut</code> …\nThe dimension type of the resulting array.\nThe dimension type of the resulting array.\nConversion trait from owning Rust types into <code>PyArray</code>.\nThe element type of resulting array.\nThe element type of resulting array.\nTrait implemented by types that can be used to index an …\nUtility trait to specify the dimensions of an array.\nConversion trait from borrowing Rust types to <code>PyArray</code>.\nDeprecated form of <code>IntoPyArray::into_pyarray_bound</code>\nConsumes <code>self</code> and moves its data into a NumPy array.\nDeprecated form of <code>ToPyArray::to_pyarray_bound</code>\nCopies the content pointed to by <code>&amp;self</code> into a newly …\nThe abbrevation used for debug formatting\nCorresponds to the <code>datetime64</code> scalar type\nCorresponds to the [<code>timedelta64</code>][scalars-datetime64] …\nThe matching NumPy datetime unit code\nRepresents the datetime units supported by NumPy\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPredefined implementors of the <code>Unit</code> trait\nAttoseconds, i.e. 10^-18 seconds\nDays, i.e. 24 hours\nFemtoseconds, i.e. 10^-15 seconds\nHours, i.e. 60 minutes\nMicroseconds, i.e. 10^-6 seconds\nMilliseconds, i.e. 10^-3 seconds\nMinutes, i.e. 60 seconds\nMonths, i.e. 30 days\nNanoseconds, i.e. 10^-9 seconds\nPicoseconds, i.e. 10^-12 seconds\nSeconds\nWeeks, i.e. 7 days\nYears, i.e. 12 months\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nLow-Level binding for Array API\nReturns whether the runtime <code>numpy</code> version is 2.0 or …\nLow-Lebel binding for NumPy C API C-objects\nLow-Level binding for UFunc API\nAll type objects exported by the NumPy API.\nA global variable which stores a ‘capsule’ pointer to …\nSee PY_ARRAY_API for more.\nChecks that <code>op</code> is an instance of <code>PyArray</code> or not.\nChecks that <code>op</code> is an exact instance of <code>PyArray</code> or not.\nReturns the argument unchanged.\nReturns the argument unchanged.\nGet a pointer of the type object assocaited with <code>ty</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nNo value.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nSome value of type <code>T</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nA global variable which stores a ‘capsule’ pointer to …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nImplementation of functionality for <code>PyArrayDescr</code>.\nImplementation of functionality for <code>PyUntypedArray</code>.\nReturns the required alignment (bytes) of this type …\nReturns a raw pointer to the underlying <code>PyArrayObject</code>.\nReturns <code>self</code> as <code>*mut PyArray_Descr</code>.\nReturns the type descriptor for the base element of …\nReturns an ASCII character indicating the byte-order of …\nReturns an ASCII character indicating the byte-order of …\nReturns an ASCII character indicating the byte-order of …\nReturns a unique ASCII character for each of the 21 …\nReturns a unique ASCII character for each of the 21 …\nReturns a unique ASCII character for each of the 21 …\nReturns the <code>dtype</code> of the array.\nReturns bit-flags describing how this type descriptor is …\nReturns the type descriptor and offset of the field with …\nReturns true if the type descriptor is a structured type.\nReturns true if the type descriptor contains any …\nReturns true if the type descriptor contains any …\nReturns true if the type descriptor contains any …\nReturns true if the type descriptor is a sub-array.\nReturns <code>self</code> as <code>*mut PyArray_Descr</code> while increasing the …\nReturns true if the type descriptor is a struct which …\nReturns true if the type descriptor is a struct which …\nReturns true if the type descriptor is a struct which …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the there are no elements in the array.\nReturns <code>true</code> if the there are no elements in the array.\nReturns <code>true</code> if the there are no elements in the array.\nReturns true if two type descriptors are equivalent.\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns <code>true</code> if the internal data of the array is …\nReturns true if type descriptor byteorder is native, or …\nReturns true if type descriptor byteorder is native, or …\nReturns true if type descriptor byteorder is native, or …\nReturns the element size of this type descriptor.\nReturns an ASCII character (one of <code>biufcmMOSUV</code>) …\nReturns an ASCII character (one of <code>biufcmMOSUV</code>) …\nReturns an ASCII character (one of <code>biufcmMOSUV</code>) …\nCalculates the total number of elements in the array.\nCalculates the total number of elements in the array.\nCalculates the total number of elements in the array.\nReturns an ordered list of field names, or <code>None</code> if there …\nReturns the number of dimensions if this type descriptor …\nReturns the number of dimensions of the array.\nReturns the number of dimensions of the array.\nReturns the number of dimensions of the array.\nReturns a unique number for each of the 21 different …\nReturns a unique number for each of the 21 different …\nReturns a unique number for each of the 21 different …\nReturns the shape of the sub-array.\nReturns a slice which contains dimmensions of the array.\nReturns a slice which contains dimmensions of the array.\nReturns a slice which contains dimmensions of the array.\nReturns a slice indicating how many bytes to advance when …\nReturns a slice indicating how many bytes to advance when …\nReturns a slice indicating how many bytes to advance when …\nReturns the array scalar corresponding to this type …")
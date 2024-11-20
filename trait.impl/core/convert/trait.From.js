(function() {
    var implementors = Object.fromEntries([["numpy",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"numpy/struct.PyArrayDescr.html\" title=\"struct numpy::PyArrayDescr\">PyArrayDescr</a>&gt; for Py&lt;<a class=\"struct\" href=\"numpy/struct.PyArrayDescr.html\" title=\"struct numpy::PyArrayDescr\">PyArrayDescr</a>&gt;"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"numpy/struct.PyUntypedArray.html\" title=\"struct numpy::PyUntypedArray\">PyUntypedArray</a>&gt; for Py&lt;<a class=\"struct\" href=\"numpy/struct.PyUntypedArray.html\" title=\"struct numpy::PyUntypedArray\">PyUntypedArray</a>&gt;"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"numpy/enum.BorrowError.html\" title=\"enum numpy::BorrowError\">BorrowError</a>&gt; for PyErr"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"numpy/struct.FromVecError.html\" title=\"struct numpy::FromVecError\">FromVecError</a>&gt; for PyErr"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"numpy/struct.NotContiguousError.html\" title=\"struct numpy::NotContiguousError\">NotContiguousError</a>&gt; for PyErr"],["impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;'a <a class=\"struct\" href=\"numpy/struct.PyArrayDescr.html\" title=\"struct numpy::PyArrayDescr\">PyArrayDescr</a>&gt; for &amp;'a PyAny"],["impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;'a <a class=\"struct\" href=\"numpy/struct.PyUntypedArray.html\" title=\"struct numpy::PyUntypedArray\">PyUntypedArray</a>&gt; for &amp;'a PyAny"],["impl&lt;'a, T, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;'a <a class=\"struct\" href=\"numpy/array/struct.PyArray.html\" title=\"struct numpy::array::PyArray\">PyArray</a>&lt;T, D&gt;&gt; for &amp;'a PyAny"],["impl&lt;'py, T, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"numpy/borrow/struct.PyReadwriteArray.html\" title=\"struct numpy::borrow::PyReadwriteArray\">PyReadwriteArray</a>&lt;'py, T, D&gt;&gt; for <a class=\"struct\" href=\"numpy/borrow/struct.PyReadonlyArray.html\" title=\"struct numpy::borrow::PyReadonlyArray\">PyReadonlyArray</a>&lt;'py, T, D&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"numpy/trait.Element.html\" title=\"trait numpy::Element\">Element</a>,\n    D: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/dimension/dimension_trait/trait.Dimension.html\" title=\"trait ndarray::dimension::dimension_trait::Dimension\">Dimension</a>,</div>"],["impl&lt;T, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;&amp;<a class=\"struct\" href=\"numpy/array/struct.PyArray.html\" title=\"struct numpy::array::PyArray\">PyArray</a>&lt;T, D&gt;&gt; for Py&lt;<a class=\"struct\" href=\"numpy/array/struct.PyArray.html\" title=\"struct numpy::array::PyArray\">PyArray</a>&lt;T, D&gt;&gt;"],["impl&lt;U: <a class=\"trait\" href=\"numpy/datetime/trait.Unit.html\" title=\"trait numpy::datetime::Unit\">Unit</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.i64.html\">i64</a>&gt; for <a class=\"struct\" href=\"numpy/datetime/struct.Datetime.html\" title=\"struct numpy::datetime::Datetime\">Datetime</a>&lt;U&gt;"],["impl&lt;U: <a class=\"trait\" href=\"numpy/datetime/trait.Unit.html\" title=\"trait numpy::datetime::Unit\">Unit</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.i64.html\">i64</a>&gt; for <a class=\"struct\" href=\"numpy/datetime/struct.Timedelta.html\" title=\"struct numpy::datetime::Timedelta\">Timedelta</a>&lt;U&gt;"],["impl&lt;U: <a class=\"trait\" href=\"numpy/datetime/trait.Unit.html\" title=\"trait numpy::datetime::Unit\">Unit</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"numpy/datetime/struct.Datetime.html\" title=\"struct numpy::datetime::Datetime\">Datetime</a>&lt;U&gt;&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.i64.html\">i64</a>"],["impl&lt;U: <a class=\"trait\" href=\"numpy/datetime/trait.Unit.html\" title=\"trait numpy::datetime::Unit\">Unit</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"numpy/datetime/struct.Timedelta.html\" title=\"struct numpy::datetime::Timedelta\">Timedelta</a>&lt;U&gt;&gt; for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.i64.html\">i64</a>"],["impl&lt;const N: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.u32.html\">u32</a>; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.array.html\">N</a>]&gt; for <a class=\"struct\" href=\"numpy/struct.PyFixedUnicode.html\" title=\"struct numpy::PyFixedUnicode\">PyFixedUnicode</a>&lt;N&gt;"],["impl&lt;const N: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.usize.html\">usize</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.82.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.u8.html\">u8</a>; <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.82.0/std/primitive.array.html\">N</a>]&gt; for <a class=\"struct\" href=\"numpy/struct.PyFixedString.html\" title=\"struct numpy::PyFixedString\">PyFixedString</a>&lt;N&gt;"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[7007]}
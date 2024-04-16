(function() {var type_impls = {
"numpy":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Element-for-f32\" class=\"impl\"><a class=\"src rightside\" href=\"src/numpy/dtype.rs.html#765\">source</a><a href=\"#impl-Element-for-f32\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"numpy/trait.Element.html\" title=\"trait numpy::Element\">Element</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.77.1/std/primitive.f32.html\">f32</a></h3></section></summary><div class=\"impl-items\"><details class=\"toggle\" open><summary><section id=\"associatedconstant.IS_COPY\" class=\"associatedconstant trait-impl\"><a class=\"src rightside\" href=\"src/numpy/dtype.rs.html#765\">source</a><a href=\"#associatedconstant.IS_COPY\" class=\"anchor\">§</a><h4 class=\"code-header\">const <a href=\"numpy/trait.Element.html#associatedconstant.IS_COPY\" class=\"constant\">IS_COPY</a>: <a class=\"primitive\" href=\"https://doc.rust-lang.org/1.77.1/std/primitive.bool.html\">bool</a> = true</h4></section></summary><div class='docblock'>Flag that indicates whether this type is trivially copyable. <a href=\"numpy/trait.Element.html#associatedconstant.IS_COPY\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.get_dtype_bound\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numpy/dtype.rs.html#765\">source</a><a href=\"#method.get_dtype_bound\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"numpy/trait.Element.html#tymethod.get_dtype_bound\" class=\"fn\">get_dtype_bound</a>(py: Python&lt;'_&gt;) -&gt; Bound&lt;'_, <a class=\"struct\" href=\"numpy/struct.PyArrayDescr.html\" title=\"struct numpy::PyArrayDescr\">PyArrayDescr</a>&gt;</h4></section></summary><div class='docblock'>Returns the associated type descriptor (“dtype”) for the given element type.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.get_dtype\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numpy/dtype.rs.html#688-690\">source</a><a href=\"#method.get_dtype\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"numpy/trait.Element.html#method.get_dtype\" class=\"fn\">get_dtype</a>&lt;'py&gt;(py: Python&lt;'py&gt;) -&gt; &amp;'py <a class=\"struct\" href=\"numpy/struct.PyArrayDescr.html\" title=\"struct numpy::PyArrayDescr\">PyArrayDescr</a></h4></section></summary><span class=\"item-info\"><div class=\"stab deprecated\"><span class=\"emoji\">👎</span><span>Deprecated since 0.21.0: This will be replaced by <code>get_dtype_bound</code> in the future.</span></div></span><div class='docblock'>Returns the associated type descriptor (“dtype”) for the given element type.</div></details></div></details>","Element","numpy::npyffi::types::npy_float","numpy::npyffi::types::npy_float32"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()
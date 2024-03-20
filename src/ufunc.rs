//! TODO

use std::ffi::CString;
use std::mem::{size_of, MaybeUninit};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::null_mut;

use ndarray::{ArrayView1, ArrayViewMut1, Axis, Dim, Ix1, ShapeBuilder, StrideShape};
use pyo3::{PyAny, PyResult, Python};

use crate::{
    dtype::Element,
    npyffi::{flags, npy_intp, objects::PyUFuncGenericFunction, ufunc::PY_UFUNC_API},
};

/// TODO
#[repr(i32)]
#[derive(Debug)]
pub enum Identity {
    /// UFunc has unit of 0, and the order of operations can be reordered
    /// This case allows reduction with multiple axes at once.
    Zero = flags::NPY_UFUNC_ZERO,
    /// UFunc has unit of 1, and the order of operations can be reordered
    /// This case allows reduction with multiple axes at once.
    One = flags::NPY_UFUNC_ONE,
    /// UFunc has unit of -1, and the order of operations can be reordered
    /// This case allows reduction with multiple axes at once. Intended for bitwise_and reduction.
    MinusOne = flags::NPY_UFUNC_MINUS_ONE,
    /// UFunc has no unit, and the order of operations cannot be reordered.
    /// This case does not allow reduction with multiple axes at once.
    None = flags::NPY_UFUNC_NONE,
    /// UFunc has no unit, and the order of operations can be reordered
    /// This case allows reduction with multiple axes at once.
    ReorderableNone = flags::NPY_UFUNC_REORDERABLE_NONE,
    /// UFunc unit is an identity_value, and the order of operations can be reordered
    /// This case allows reduction with multiple axes at once.
    IdentityValue = flags::NPY_UFUNC_IDENTITY_VALUE,
}

/// TODO
///
/// ```
/// # #![allow(mixed_script_confusables)]
/// # use std::ffi::CString;
/// #
/// use pyo3::{py_run, Python};
/// use ndarray::{azip, ArrayView1, ArrayViewMut1};
/// use numpy::ufunc::{from_func, Identity};
///
/// Python::with_gil(|py| {
///    let logit = |[p]: [ArrayView1<'_, f64>; 1], [α]: [ArrayViewMut1<'_, f64>; 1]| {
///        azip!((p in p, α in α) {
///            let mut tmp = *p;
///            tmp /= 1.0 - tmp;
///            *α = tmp.ln();
///        });
///    };
///
///    let logit =
///        from_func(py, CString::new("logit").unwrap(), Identity::None, logit).unwrap();
///
///    py_run!(py, logit, "assert logit(0.5) == 0.0");
///
///    let np = py.import("numpy").unwrap();
///
///    py_run!(py, logit np, "assert (logit(np.full(100, 0.5)) == np.zeros(100)).all()");
/// });
/// ```
///
/// ```
/// # #![allow(mixed_script_confusables)]
/// # use std::ffi::CString;
/// #
/// use pyo3::{py_run, Python};
/// use ndarray::{azip, ArrayView1, ArrayViewMut1};
/// use numpy::ufunc::{from_func, Identity};
///
/// Python::with_gil(|py| {
///     let cart_to_polar = |[x, y]: [ArrayView1<'_, f64>; 2], [r, φ]: [ArrayViewMut1<'_, f64>; 2]| {
///         azip!((&x in x, &y in y, r in r, φ in φ) {
///             *r = f64::hypot(x, y);
///             *φ = f64::atan2(x, y);
///         });
///     };
///
///     let cart_to_polar = from_func(
///         py,
///         CString::new("cart_to_polar").unwrap(),
///         Identity::None,
///         cart_to_polar,
///     )
///     .unwrap();
///
///     let np = py.import("numpy").unwrap();
///
///     py_run!(py, cart_to_polar np, "np.testing.assert_array_almost_equal(cart_to_polar(3.0, 4.0), (5.0, 0.643501))");
///
///     py_run!(py, cart_to_polar np, "np.testing.assert_array_almost_equal(cart_to_polar(np.full((10, 10), 3.0), np.full((10, 10), 4.0))[0], np.full((10, 10), 5.0))");
///     py_run!(py, cart_to_polar np, "np.testing.assert_array_almost_equal(cart_to_polar(np.full((10, 10), 3.0), np.full((10, 10), 4.0))[1], np.full((10, 10), 0.643501))");
/// });
/// ```
pub fn from_func<'py, T, F, const NIN: usize, const NOUT: usize>(
    py: Python<'py>,
    name: CString,
    identity: Identity,
    func: F,
) -> PyResult<&'py PyAny>
where
    T: Element,
    F: Fn([ArrayView1<'_, T>; NIN], [ArrayViewMut1<'_, T>; NOUT]) + 'static,
{
    let wrap_func = [Some(wrap_func::<T, F, NIN, NOUT> as _)];

    let r#type = T::get_npy_type().expect("universal function only work for built-in types");

    let inputs = [r#type as _; NIN];
    let outputs = [r#type as _; NOUT];

    let data = Data {
        func,
        wrap_func,
        name,
        inputs,
        outputs,
    };

    let data = Box::leak(Box::new(data));

    unsafe {
        py.from_owned_ptr_or_err(PY_UFUNC_API.PyUFunc_FromFuncAndData(
            py,
            data.wrap_func.as_mut_ptr(),
            data as *mut Data<F, NIN, NOUT> as *mut c_void as *mut *mut c_void,
            data.inputs.as_mut_ptr(),
            /* ntypes = */ 1,
            NIN as c_int,
            NOUT as c_int,
            identity as c_int,
            data.name.as_ptr(),
            /* doc = */ null_mut(),
            /* unused = */ 0,
        ))
    }
}

#[repr(C)]
struct Data<F, const NIN: usize, const NOUT: usize> {
    func: F,
    wrap_func: [PyUFuncGenericFunction; 1],
    name: CString,
    inputs: [c_char; NIN],
    outputs: [c_char; NOUT],
}

unsafe extern "C" fn wrap_func<T, F, const NIN: usize, const NOUT: usize>(
    args: *mut *mut c_char,
    dims: *mut npy_intp,
    steps: *mut npy_intp,
    data: *mut c_void,
) where
    F: Fn([ArrayView1<'_, T>; NIN], [ArrayViewMut1<'_, T>; NOUT]),
{
    // TODO: Check aliasing requirements using the `borrow` module.

    let mut inputs = MaybeUninit::<[ArrayView1<'_, T>; NIN]>::uninit();
    let inputs_ptr = inputs.as_mut_ptr() as *mut ArrayView1<'_, T>;

    for i in 0..NIN {
        let (ptr, shape, invert) = unpack_arg(args, dims, steps, i);

        let mut input = ArrayView1::from_shape_ptr(shape, ptr);
        if invert {
            input.invert_axis(Axis(0));
        }
        inputs_ptr.add(i).write(input);
    }

    let mut outputs = MaybeUninit::<[ArrayViewMut1<'_, T>; NOUT]>::uninit();
    let outputs_ptr = outputs.as_mut_ptr() as *mut ArrayViewMut1<'_, T>;

    for i in 0..NOUT {
        let (ptr, shape, invert) = unpack_arg(args, dims, steps, NIN + i);

        let mut output = ArrayViewMut1::from_shape_ptr(shape, ptr);
        if invert {
            output.invert_axis(Axis(0));
        }
        outputs_ptr.add(i).write(output);
    }

    let data = &*(data as *mut Data<F, NIN, NOUT>);
    (data.func)(inputs.assume_init(), outputs.assume_init());
}

unsafe fn unpack_arg<T>(
    args: *mut *mut c_char,
    dims: *mut npy_intp,
    steps: *mut npy_intp,
    i: usize,
) -> (*mut T, StrideShape<Ix1>, bool) {
    let dim = Dim([*dims as usize]);
    let itemsize = size_of::<T>();

    let mut ptr = *args.add(i);
    let mut invert = false;

    let step = *steps.add(i);

    let step = if step >= 0 {
        Dim([step as usize / itemsize])
    } else {
        ptr = ptr.offset(step * (*dims - 1));
        invert = true;

        Dim([(-step) as usize / itemsize])
    };

    (ptr as *mut T, dim.strides(step), invert)
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::azip;
    use pyo3::py_run;

    #[test]
    fn from_func_handles_negative_strides() {
        Python::with_gil(|py| {
            let negate = from_func(
                py,
                CString::new("negate").unwrap(),
                Identity::None,
                |[x]: [ArrayView1<'_, f64>; 1], [y]: [ArrayViewMut1<'_, f64>; 1]| {
                    azip!((x in x, y in y) *y = -x);
                },
            )
            .unwrap();

            let np = py.import("numpy").unwrap();

            py_run!(py, negate np, "assert (negate(np.linspace(1.0, 10.0, 10)[::-1]) == np.linspace(-10.0, -1.0, 10)).all()");
        });
    }
}

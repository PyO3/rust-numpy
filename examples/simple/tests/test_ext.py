import numpy as np
from rust_ext import axpy, conj, mult, extract


def test_axpy():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 3.0, 3.0])
    z = axpy(3.0, x, y)
    np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0]))


def test_mult():
    x = np.array([1.0, 2.0, 3.0])
    mult(3.0, x)
    np.testing.assert_array_almost_equal(x, np.array([3.0, 6.0, 9.0]))


def test_conj():
    x = np.array([1.0 + 2j, 2.0 + 3j, 3.0 + 4j])
    np.testing.assert_array_almost_equal(conj(x), np.conj(x))


def test_extract():
    x = np.arange(5.0)
    d = {"x": x}
    np.testing.assert_almost_equal(extract(d), 10.0)

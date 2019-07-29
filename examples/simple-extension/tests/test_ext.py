import numpy as np
from rust_ext import axpy, mult, padd


def test_axpy():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 3.0, 3.0])
    z = axpy(3.0, x, y)
    np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0]))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    z = axpy(3.0, x, y)
    np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0, 15.0]))


def test_mult():
    x = np.array([1.0, 2.0, 3.0])
    mult(3.0, x)
    np.testing.assert_array_almost_equal(x, np.array([3.0, 6.0, 9.0]))


def test_padd():
    x = np.zeros((10, 1000))
    y = np.ones((10, 1000))
    padd(x, y)
    np.testing.assert_array_almost_equal(x, np.ones((10, 1000)))

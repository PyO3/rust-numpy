import numpy as np
from rust_ext import axpy, mult
import unittest

class TestExt(unittest.TestCase):
    """Test class for rust functions
    """

    def test_axpy(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 3.0, 3.0])
        z = axpy(3.0, x, y)
        np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0]))

    def test_mult(self):
        x = np.array([1.0, 2.0, 3.0])
        mult(3.0, x)
        np.testing.assert_array_almost_equal(x, np.array([3.0, 6.0, 9.0]))


if __name__ == "__main__":
    unittest.main()

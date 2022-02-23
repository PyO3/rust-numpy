import numpy as np
import rust_linalg


def test_inv():
    x = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.float64,
    )
    y = rust_linalg.inv(x)
    np.testing.assert_array_almost_equal(y, np.linalg.inv(x))

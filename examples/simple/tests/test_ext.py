import numpy as np
from rust_ext import (
    head,
    axpy,
    conj,
    mult,
    extract,
    add_minutes_to_seconds,
    polymorphic_add,
)


def test_head():
    x = np.array(["first", None, 42])
    first = head(x)
    assert first == "first"


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


def test_add_minutes_to_seconds():
    x = np.array([10, 20, 30], dtype="timedelta64[s]")
    y = np.array([1, 2, 3], dtype="timedelta64[m]")

    add_minutes_to_seconds(x, y)

    assert np.all(x == np.array([70, 140, 210], dtype="timedelta64[s]"))


def test_polymorphic_add():
    x = np.array([1.0, 2.0, 3.0], dtype=np.double)
    y = np.array([3.0, 3.0, 3.0], dtype=np.double)
    z = polymorphic_add(x, y)
    np.testing.assert_array_almost_equal(z, np.array([4.0, 5.0, 6.0], dtype=np.double))

    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([3, 3, 3], dtype=np.int64)
    z = polymorphic_add(x, y)
    assert np.all(z == np.array([4, 5, 6], dtype=np.int64))

    x = np.array([1.0, 2.0, 3.0], dtype=np.double)
    y = np.array([3, 3, 3], dtype=np.int64)
    z = polymorphic_add(x, y)
    np.testing.assert_array_almost_equal(z, np.array([4.0, 5.0, 6.0], dtype=np.double))

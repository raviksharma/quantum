import numpy as np

from base import zero, one
from gates import X, H, CNOT


def test_X():
    assert(np.array_equal(X(zero), one))
    assert(np.array_equal(X(one), zero))
    assert(np.array_equal(X(X(zero)), zero))
    assert(np.array_equal(X(X(one)), one))

    s = np.array([[.6],
                  [.8]], dtype='complex')
    assert(np.array_equal(X(s), np.array([[.8],
                                          [.6]], dtype=np.complex)))
    assert(np.array_equal(X(X(s)), s))


def test_H():
    assert(np.allclose(H(H(zero)), zero))
    assert(np.allclose(H(H(one)), one))


def test_CNOT():
    # |00>
    zz = np.array([[1],
                   [0],
                   [0],
                   [0]], dtype=np.complex)

    # |01>
    zo = np.array([[0],
                   [1],
                   [0],
                   [0]], dtype=np.complex)

    # |10>
    oz = np.array([[0],
                   [0],
                   [1],
                   [0]], dtype=np.complex)

    # |11>
    oo = np.array([[0],
                   [0],
                   [0],
                   [1]], dtype=np.complex)

    assert(np.allclose(CNOT(zz), zz))
    assert(np.allclose(CNOT(zo), zo))
    assert(np.allclose(CNOT(oz), oo))
    assert(np.allclose(CNOT(oo), oz))

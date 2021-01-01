import math
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
    h0 = (zero+one)/math.sqrt(2)
    h1 = (zero-one)/math.sqrt(2)
    assert(np.allclose(H(zero), h0))
    assert(np.allclose(H(one), h1))

    alpha = 0.6
    beta = 0.8
    s = np.array([[alpha],
                  [beta]], dtype='complex')
    assert(np.allclose(H(s), alpha*h0 + beta*h1))
    assert(np.allclose(H(s), (((alpha+beta)*zero)/math.sqrt(2)) + (((alpha-beta)*one)/math.sqrt(2))))

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

import math
import numpy as np

from base import zero, one, \
                 zz, zo, oz, oo, \
                 zzz, zzo, zoz, zoo, ozz, ozo, ooz, ooo
from gates import X, Z, H, CNOT, SWAP, CCNOT, CSWAP


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


def test_Z():
    assert(np.allclose(H(Z(H(zero))), X(zero)))
    assert(np.allclose(H(Z(H(one))), X(one)))
    s = np.array([[.6],
                  [.8]], dtype='complex')
    assert(np.allclose(H(Z(H(s))), X(s)))

    assert(np.allclose(H(X(H(zero))), Z(zero)))
    assert(np.allclose(H(X(H(one))), Z(one)))
    s = np.array([[.6],
                  [.8]], dtype='complex')
    assert(np.allclose(H(X(H(s))), Z(s)))

def test_CNOT():
    assert(np.allclose(CNOT(zz), zz))
    assert(np.allclose(CNOT(zo), zo))
    assert(np.allclose(CNOT(oz), oo))
    assert(np.allclose(CNOT(oo), oz))


def test_SWAP():
    assert(np.allclose(SWAP(zz), zz))
    assert(np.allclose(SWAP(zo), oz))
    assert(np.allclose(SWAP(oz), zo))
    assert(np.allclose(SWAP(oo), oo))


def test_CCNOT():
    assert(np.allclose(CCNOT(zzz), zzz))
    assert(np.allclose(CCNOT(zzo), zzo))
    assert(np.allclose(CCNOT(zoz), zoz))
    assert(np.allclose(CCNOT(zoo), zoo))
    assert(np.allclose(CCNOT(ozz), ozz))
    assert(np.allclose(CCNOT(ozo), ozo))
    assert(np.allclose(CCNOT(ooz), ooo))
    assert(np.allclose(CCNOT(ooo), ooz))


def test_CSWAP():
    assert(np.allclose(CSWAP(zzz), zzz))
    assert(np.allclose(CSWAP(zzo), zzo))
    assert(np.allclose(CSWAP(zoz), zoz))
    assert(np.allclose(CSWAP(zoo), zoo))
    assert(np.allclose(CSWAP(ozz), ozz))
    assert(np.allclose(CSWAP(ozo), ooz))
    assert(np.allclose(CSWAP(ooz), ozo))
    assert(np.allclose(CSWAP(ooo), ooo))

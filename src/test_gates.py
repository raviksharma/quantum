from math import e, pi, sqrt
import numpy as np

from base import zero, one, \
                 zz, zo, oz, oo, \
                 zzz, zzo, zoz, zoo, ozz, ozo, ooz, ooo
from gates import X, Y, Z, RZ, S, T, H, K, H2, \
                  CNOT, SWAP, CCNOT, CSWAP, U3, U2, U1


alpha = 0.6
beta = 0.8
s = np.array([[alpha],
              [beta]], dtype='complex')

def test_X():
    assert(np.array_equal(X(zero), one))
    assert(np.array_equal(X(one), zero))
    assert(np.array_equal(X(X(zero)), zero))
    assert(np.array_equal(X(X(one)), one))

    assert(np.array_equal(X(s), np.array([[beta],
                                          [alpha]], dtype=complex)))
    assert(np.array_equal(X(X(s)), s))

def test_H():
    h0 = (zero+one)/sqrt(2)
    h1 = (zero-one)/sqrt(2)
    # H(zero), also known as |+>
    assert(np.allclose(H(zero), h0))
    # H(one), also known as |->
    assert(np.allclose(H(one), h1))

    assert(np.allclose(H(s), alpha*h0 + beta*h1))
    assert(np.allclose(H(s), (((alpha+beta)*zero)/sqrt(2)) + (((alpha-beta)*one)/sqrt(2))))

    assert(np.allclose(H(H(zero)), zero))
    assert(np.allclose(H(H(one)), one))

def test_H2():
    # H2|00> = (|00> + |01> + |10> + |11>)/2
    assert(np.allclose(H2(zz),(zz+zo+oz+oo)/2))
    assert(np.allclose(H2(zo),(zz-zo+oz-oo)/2))
    assert(np.allclose(H2(oz),(zz+zo-oz-oo)/2))
    assert(np.allclose(H2(oo),(zz-zo-oz+oo)/2))

    assert(np.allclose(CNOT(H2(zz)),H2(zz)))
    assert(np.allclose(CNOT(H2(zo)),H2(oo)))
    assert(np.allclose(CNOT(H2(oz)),H2(oz)))
    assert(np.allclose(CNOT(H2(oo)),H2(zo)))

def test_K():
    assert(np.allclose(K((zz+oo)/sqrt(2)),(zz+zo+oz-oo)/2))

def test_Z():
    assert(np.allclose(Z(Z(s)), s))

    assert(np.allclose(H(Z(H(zero))), X(zero)))
    assert(np.allclose(H(Z(H(one))), X(one)))
    assert(np.allclose(H(Z(H(s))), X(s)))

    assert(np.allclose(H(X(H(zero))), Z(zero)))
    assert(np.allclose(H(X(H(one))), Z(one)))
    assert(np.allclose(H(X(H(s))), Z(s)))

def test_RZ():
    assert(np.allclose(Z(zero), RZ(pi, zero)))
    assert(np.allclose(Z(one), RZ(pi, one)))
    assert(np.allclose(Z(s), RZ(pi, s)))

def test_S():
    assert(np.allclose(S(S(s)), Z(s)))
    assert(np.allclose(S(S(zero)), Z(zero)))
    assert(np.allclose(S(S(one)), Z(one)))

def test_T():
    assert(np.allclose(T(T(s)), S(s)))
    assert(np.allclose(T(T(zero)), S(zero)))
    assert(np.allclose(T(T(one)), S(one)))

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

def test_U1():
    assert(np.allclose(U1(pi, zero), RZ(pi, zero)))
    assert(np.allclose(U1(2*pi, zero), RZ(2*pi, zero)))
    assert(np.allclose(U1((3*pi)/2, zero), RZ((3*pi)/2, zero)))
    assert(np.allclose(U1(pi/2, one), RZ(pi/2, one)))
    assert(np.allclose(U1(pi/4, s), RZ(pi/4, s)))

def test_U2():
    assert(np.allclose(U2(0, pi, zero), H(zero)))
    assert(np.allclose(U2(0, pi, one), H(one)))
    assert(np.allclose(U2(0, pi, s), H(s)))

def test_U3():
    assert(np.allclose(U3(pi, 0, pi, zero), X(zero)))
    assert(np.allclose(U3(pi, 0, pi, one), X(one)))
    assert(np.allclose(U3(pi, 0, pi, s), X(s)))

    assert(np.allclose(U3(pi, pi/2, pi/2, zero), Y(zero)))
    assert(np.allclose(U3(pi, pi/2, pi/2, one), Y(one)))
    assert(np.allclose(U3(pi, pi/2, pi/2, s), Y(s)))

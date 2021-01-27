import numpy as np

# computational basis state 0 - |0> (ket notation)
zero = np.array([[1],
                 [0]], dtype=np.complex)

# computational basis state 1 - |1>
one = np.array([[0],
                [1]], dtype=np.complex)

# general state
# |ψ> = α |0> + β |1>,
# where α, β ∈ Complex,
# and |α|**2 + |β|**2 = 1, normalization constraint

# |ab> = |a> kronecker_product |b>
# a kronecker product is used to combine quantum states
# https://en.wikipedia.org/wiki/Kronecker_product

# |00> / vector representation
zz = np.array([[1],
	       [0],
	       [0],
	       [0]], dtype=np.complex)

# or
zz = np.kron(zero, zero)

# |01>
zo = np.kron(zero, one)

# |10>
oz = np.kron(one, zero)

# |11>
oo = np.kron(one, one)

# |000>
zzz = np.kron(zero, np.kron(zero, zero))

# |001>
zzo = np.kron(zero, np.kron(zero, one))

# |010>
zoz = np.kron(zero, np.kron(one, zero))

# |011>
zoo = np.kron(zero, np.kron(one, one))

# |100>
ozz = np.kron(one, np.kron(zero, zero))

# |101>
ozo = np.kron(one, np.kron(zero, one))

# |110>
ooz = np.kron(one, np.kron(one, zero))

# |111>
ooo = np.kron(one, np.kron(one, one))

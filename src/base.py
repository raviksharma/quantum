import numpy as np

#  computational basis state 0 - |0> (ket notation)
zero = np.array([[1],
                 [0]], dtype=np.complex)

#  computational basis state 1 - |1>
one = np.array([[0],
                [1]], dtype=np.complex)

# general state
# |ψ> = α |0> + β |1>,
# where α, β ∈ Complex,
# and |α|**2 + |β|**2 = 1, normalization constraint

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

# |000>
zzz = np.array([[1],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0]], dtype=np.complex)

# |001>
zzo = np.array([[0],
	        [1],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0]], dtype=np.complex)

# |010>
zoz = np.array([[0],
	        [0],
	        [1],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0]], dtype=np.complex)

# |011>
zoo = np.array([[0],
	        [0],
	        [0],
	        [1],
	        [0],
	        [0],
	        [0],
	        [0]], dtype=np.complex)

# |100>
ozz = np.array([[0],
	        [0],
	        [0],
	        [0],
	        [1],
	        [0],
	        [0],
	        [0]], dtype=np.complex)

# |101>
ozo = np.array([[0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [1],
	        [0],
	        [0]], dtype=np.complex)

# |110>
ooz = np.array([[0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [1],
	        [0]], dtype=np.complex)

# |111>
ooo = np.array([[0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [0],
	        [1]], dtype=np.complex)

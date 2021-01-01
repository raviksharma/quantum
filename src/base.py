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

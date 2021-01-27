from math import e, pi, sqrt, cos, sin
import numpy as np


not_vector = np.array([[0, 1],
                       [1, 0]], dtype=np.complex)

y_vector = np.array([[0, -1j],
                     [1j, 0]], dtype=np.complex)

h_vector = (1/sqrt(2)) * np.array([[1, 1],
                                   [1, -1]], dtype=np.complex)

#  Pauli-X / NOT gate / Half turn
def X(state):
    """Applies the NOT gate.

    state -- state of a single qubit
    """
    return not_vector @ state

#  Hadamard gate / Half turn
def H(state):
    """Applies the Hadamard gate.

    state -- state of a single qubit
    """
    return h_vector @ state

#  Hadamard transform - Hadamard gate (H) applied in parallel on 2 qubits
def H2(joint_state):
    """Applies the Hadamard transform.

    joint_state -- joint state of 2 qubits
    """
    return np.kron(h_vector, h_vector) @ joint_state

#  Pauli-Y / Y gate / Half turn
def Y(state):
    """Applies the Y gate.

    state -- state of a single qubit
    """
    return y_vector @ state

#  Pauli-Z / Z gate / Half turn
def Z(state):
    """Applies the Z gate.

    state -- state of a single qubit
    """
    z_vector = np.array([[1, 0],
                         [0, -1]], dtype=np.complex)
    return z_vector @ state

#  Phase shift / RZ gate
def RZ(phi, state):
    """Applies the RZ gate.

    state -- state of a single qubit
    """
    rz_vector = np.array([[1, 0],
                          [0, e ** (phi * 1j)]], dtype=np.complex)
    return rz_vector @ state

#  S gate / sqrt(Z) / Quarter turn
def S(state):
    """Applies the S gate.

    state -- state of a single qubit
    """
    return RZ(pi/2, state)

#  T gate / sqrt(S) / Eighth turn
def T(state):
    """Applies the T gate.

    state -- state of a single qubit
    """
    return RZ(pi/4, state)


# Controlled NOT / CNOT / CX gate
def CNOT(joint_state):
    """Applies the CNOT gate.

    joint_state -- joint state of 2 qubits
    """
    cnot_vector = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=np.complex)
    return cnot_vector @ joint_state

#  Controlled Phase shift / CPHASE gate
def CPHASE(phi, joint_state):
    """Applies the CPHASE gate.

    joint_state -- joint state of 2 qubits
    """
    cr_vector = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, e ** (phi * 1j)]], dtype=np.complex)
    return cr_vector @ joint_state


# Controlled Z / CZ gate
def CZ(joint_state):
    """Applies the CZ gate.

    joint_state -- joint state of 2 qubits
    """
    cz_vector = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, -1]], dtype=np.complex)
    return cz_vector @ joint_state

# Controlled U / CU gate
def CU(U, joint_state):
    """Applies the CU gate.

    U           -- Pauli matrices
    joint_state -- joint state of 2 qubits
    """
    cu_vector = np.identity(4, dtype=np.complex)
    cu_vector[2:4, 2:4] = U
    return cu_vector @ joint_state

# Controlled X / CX gate / same as CNOT
def CX(joint_state):
    """Applies the CX gate.

    joint_state -- joint state of 2 qubits
    """
    return CU(not_vector, joint_state)

# Controlled Y / CY gate
def CY(joint_state):
    """Applies the CY gate.

    joint_state -- joint state of 2 qubits
    """
    return CU(y_vector, joint_state)

# SWAP gate
def SWAP(joint_state):
    """Applies the SWAP gate.

    joint_state -- joint state of 2 qubits
    """
    swap_vector = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.complex)
    return swap_vector @ joint_state

# Toffoli / CCNOT gate
def CCNOT(joint_state):
    """Applies the CCNOT gate.

    joint_state -- joint state of 3 qubits
    """
    ccnot_vector = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.complex)
    return ccnot_vector @ joint_state

# Fredkin / CSWAP gate
def CSWAP(joint_state):
    """Applies the CSWAP gate.

    joint_state -- joint state of 3 qubits
    """
    cswap_vector = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.complex)
    return cswap_vector @ joint_state

# Universal quantum gates for single qubit
def U3(_theta, _phi, _lambda, state):
    """Applies the U3 gate.

    state   -- state of a single qubit
    """
    u3_vector = np.array([[cos(_theta/2), -((e ** (_lambda * 1j) * (sin(_theta/2))))],
                         [((e ** (_phi * 1j) * (sin(_theta/2)))), ((e ** ((_lambda * 1j) + (_phi * 1j)) * (cos(_theta/2))))]], dtype=np.complex)
    return u3_vector @ state

def U2(_phi, _lambda, state):
    return U3(pi/2, _phi, _lambda, state)

def U1(_lambda, state):
    return U3(0, 0, _lambda, state)

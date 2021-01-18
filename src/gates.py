from math import e, pi, sqrt
import numpy as np


#  Pauli-X / NOT gate / Half turn
def X(state):
    not_vector = np.array([[0, 1],
                           [1, 0]], dtype=np.complex)
    return not_vector @ state


#  Hadamard gate / Half turn
def H(state):
    h_vector = (1/sqrt(2)) * np.array([[1, 1],
                                       [1, -1]], dtype=np.complex)
    return h_vector @ state


#  Pauli-Y / Y gate / Half turn
def Y(state):
    y_vector = np.array([[0, -1j],
                         [1j, 0]], dtype=np.complex)
    return y_vector @ state


#  Pauli-Z / Z gate / Half turn
def Z(state):
    """Applies Z gate.

    state -- state of a single qubit
    """
    z_vector = np.array([[1, 0],
                         [0, -1]], dtype=np.complex)
    return z_vector @ state

#  Phase shift / RZ gate
def RZ(phi, state):
    """Applies RZ gate.

    state -- state of a single qubit
    """
    rz_vector = np.array([[1, 0],
                          [0, e ** (phi * 1j)]], dtype=np.complex)
    return rz_vector @ state

#  S gate / sqrt(Z) / Quarter turn
def S(state):
    """Applies S gate.

    state -- state of a single qubit
    """
    return RZ(pi/2, state)

#  T gate / sqrt(S) / Eighth turn
def T(state):
    """Applies T gate.

    state -- state of a single qubit
    """
    return RZ(pi/4, state)


# Controlled NOT / CNOT gate
def CNOT(joint_state):
    """Applies CNOT gate.

    joint_state -- joint state of 2 qubits
    """
    cnot_vector = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]], dtype=np.complex)
    return cnot_vector @ joint_state

#  Controlled Phase shift / CPHASE gate
def CPHASE(phi, joint_state):
    """Applies CPHASE gate.

    joint_state -- joint state of 2 qubits
    """
    cr_vector = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, e ** (phi * 1j)]], dtype=np.complex)
    return cr_vector @ joint_state


# Controlled Z / CZ gate
def CZ(joint_state):
    """Applies CZ gate.

    joint_state -- joint state of 2 qubits
    """
    cz_vector = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, -1]], dtype=np.complex)
    return cz_vector @ joint_state


# SWAP gate
def SWAP(joint_state):
    """Applies SWAP gate.

    joint_state -- joint state of 2 qubits
    """
    swap_vector = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.complex)
    return swap_vector @ joint_state

# Toffoli / CCNOT gate
def CCNOT(joint_state):
    """Applies CCNOT gate.

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
    """Applies CSWAP gate.

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

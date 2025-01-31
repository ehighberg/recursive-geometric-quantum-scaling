# utils/quantum_state.py

import numpy as np
from qutip import Qobj, basis, tensor

def state_zero(num_qubits=1):
    """All-|0> state."""
    if num_qubits == 1:
        return basis(2, 0)
    return tensor(*[basis(2, 0) for _ in range(num_qubits)])

def state_one(num_qubits=1):
    """All-|1> state."""
    if num_qubits == 1:
        return basis(2, 1)
    return tensor(*[basis(2, 1) for _ in range(num_qubits)])

def state_plus(num_qubits=1):
    """All-|+> = (|0>+|1>)/√2 (tensored for multiple qubits)."""
    single_plus = (basis(2, 0) + basis(2, 1)).unit()
    if num_qubits == 1:
        return single_plus
    return tensor(*[single_plus for _ in range(num_qubits)])

def state_ghz(num_qubits=3):
    """
    GHZ = (|000...> + |111...>)/√2, for >=2 qubits.
    """
    if num_qubits < 2:
        raise ValueError("GHZ requires >=2 qubits.")
    zero_ket = tensor(*[basis(2, 0) for _ in range(num_qubits)])
    one_ket = tensor(*[basis(2, 1) for _ in range(num_qubits)])
    return (zero_ket + one_ket).unit()

def state_w(num_qubits=3):
    """
    W = (|100..> + |010..> + ... )/√(num_qubits).
    """
    if num_qubits < 2:
        raise ValueError("W requires >=2 qubits.")
    if num_qubits == 0:
        raise ValueError("W requires > 0 qubits.")
    states = []
    for i in range(num_qubits):
        ket_list = []
        for j in range(num_qubits):
            ket_list.append(basis(2, 1) if j == i else basis(2, 0))
        states.append(tensor(*ket_list))
    psi = sum(states)
    if isinstance(psi, Qobj):
        return psi.unit()
    else:
        return psi

def positivity_projection(rho):
    """
    Clamps negative eigenvalues in a density matrix and re-normalizes.
    """
    evals, evecs = rho.eigenstates()
    evals_clipped = [max(ev, 0.0) for ev in evals]
    rho_fixed = 0 * rho
    for val, vec in zip(evals_clipped, evecs):
        rho_fixed += val * vec * vec.dag()

    if rho_fixed.norm() < 1e-15:
        return rho_fixed
    tr_val = rho_fixed.tr()
    if tr_val > 1e-15:
        rho_fixed /= tr_val
    return rho_fixed

# Toy states for fibonacci anyons:
def fib_anyon_state_2d():
    """
    2D subspace for 3 anyons => dimension=2. 
    We'll do an equal superposition for demonstration.
    """
    vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    return Qobj(vec)

def fib_anyon_state_3d():
    """
    3D subspace => dimension=3, for 4 anyons total charge τ.
    We'll do a basis vector [1,0,0].
    """
    vec = np.array([1.0, 0.0, 0.0], dtype=complex)
    return Qobj(vec)

# simualtations/quantum_state.py

import numpy as np
from qutip import Qobj, basis, tensor

def state_zero(num_qubits=1):
    if num_qubits==1:
        return basis(2,0)
    return tensor(*[basis(2,0) for _ in range(num_qubits)])

def state_one(num_qubits=1):
    if num_qubits==1:
        return basis(2,1)
    return tensor(*[basis(2,1) for _ in range(num_qubits)])

def state_plus(num_qubits=1):
    """
    Single-qubit |+> = (|0> + |1>)/sqrt(2).
    For multiple qubits, tensor of |+>.
    """
    single_plus = (basis(2,0)+basis(2,1)).unit()
    if num_qubits==1:
        return single_plus
    return tensor(*[single_plus for _ in range(num_qubits)])

def state_ghz(num_qubits=3):
    if num_qubits<2:
        raise ValueError("GHZ requires >=2 qubits")
    zero_ket = tensor(*[basis(2,0) for _ in range(num_qubits)])
    one_ket  = tensor(*[basis(2,1) for _ in range(num_qubits)])
    ghz = (zero_ket + one_ket).unit()
    return ghz

def state_w(num_qubits=3):
    if num_qubits<2:
        raise ValueError("W requires >=2 qubits")
    states = []
    for i in range(num_qubits):
        ket_list=[]
        for j in range(num_qubits):
            if j==i:
                ket_list.append(basis(2,1))
            else:
                ket_list.append(basis(2,0))
        states.append(tensor(*ket_list))
    # Use reduce to maintain Qobj type during summation
    from functools import reduce
    psi = reduce(lambda x, y: x + y, states)
    return psi.unit()

def positivity_projection(rho):
    """
    Projects a density matrix onto the positive semidefinite cone, then renormalizes.
    """
    evals, evecs = rho.eigenstates()
    evals_clipped = [max(ev,0) for ev in evals]
    rho_fixed = 0*rho
    for val, vec in zip(evals_clipped, evecs):
        rho_fixed += val * (vec @ vec.dag())
    tr = rho_fixed.tr()
    if tr>1e-15:
        rho_fixed /= tr
    return rho_fixed

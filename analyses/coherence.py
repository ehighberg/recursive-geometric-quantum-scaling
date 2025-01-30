# analyses/coherence.py

"""
Coherence analysis measures:
 - L1 norm of coherence
 - Relative entropy of coherence
for a chosen "computational" basis. We'll illustrate a simple approach.
"""

import numpy as np
from qutip import Qobj

def l1_coherence(rho, dim=None):
    """
    L1 measure of coherence in computational basis:
      C_L1(ρ) = sum_{i != j} |ρ_{i,j}|
    If dim is None, assume the dimension from rho.shape.
    """
    if rho.isket or rho.isoper == False:
        raise ValueError("rho must be a density matrix (operator) for coherence measure.")
    arr = rho.full()
    if dim is None:
        dim = arr.shape[0]
    # sum off-diagonal absolute values
    off_diag_sum = 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                off_diag_sum += abs(arr[i,j])
    return off_diag_sum

def relative_entropy_coherence(rho, dim=None):
    """
    Relative entropy of coherence:
      C_{rel}(ρ) = S(diag(ρ)) - S(ρ),
    where S is von Neumann entropy, diag(ρ) is ρ with off-diagonal set to 0 in the chosen basis.
    """
    from qutip import Qobj, entropy_vn
    arr = rho.full()
    if dim is None:
        dim = arr.shape[0]

    # diag(ρ) => zero out off-diagonals
    diag_rho = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        diag_rho[i,i] = arr[i,i]

    from qutip import Qobj
    diag_rho_q = Qobj(diag_rho, dims=rho.dims)
    # S(diag(rho)) - S(rho)
    S_diag = entropy_vn(diag_rho_q)
    S_rho  = entropy_vn(rho)
    return S_diag - S_rho

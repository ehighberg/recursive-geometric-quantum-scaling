# tests/test_entropy.py

import pytest
import numpy as np
from qutip import basis, ket2dm, tensor
from analyses.entropy import (
    compute_vn_entropy, compute_linear_entropy, compute_mutual_information
)

def test_vn_entropy_pure_state():
    """
    A pure state => von Neumann entropy = 0 (base=2 or base=e => 0 either way).
    """
    psi = tensor(basis(2,0), basis(2,0))  # |00>
    rho = ket2dm(psi)
    S_rho = compute_vn_entropy(rho, base=2)
    assert abs(S_rho - 0.0) < 1e-10

def test_linear_entropy_pure_state():
    """
    A pure state => linear entropy = 0.
    """
    psi = tensor(basis(2,0), basis(2,0))
    rho = ket2dm(psi)
    l_ent = compute_linear_entropy(rho)
    assert abs(l_ent) < 1e-14

def test_mutual_information_bell():
    """
    For a 2-qubit Bell state, 1st qubit = A, 2nd qubit = B => 
    S(A)=1, S(B)=1, S(AB)=0 => I(A:B)=2 in base=2 logs
    But let's confirm with the default base in QuTiP (natural logs).
    Then S(A)= ln(2), S(B)= ln(2), S(AB)=0 => I= 2 ln(2)= ~1.386
    """
    from qutip import ket2dm
    psi_bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    rho_bell = ket2dm(psi_bell)

    # default is natural log => S(A)= ln(2)=0.693..., S(B)= ln(2)=0.693..., S(AB)=0 => sum=1.386
    I_AB = compute_mutual_information(rho_bell, subsysA=0, subsysB=1, dims=[2,2])
    assert abs(I_AB - 2*np.log(2)) < 1e-4, f"Mutual info ~ 1.386, got {I_AB}"

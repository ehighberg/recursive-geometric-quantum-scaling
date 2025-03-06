# tests/test_entropy.py

import pytest
import numpy as np
from qutip import basis, ket2dm, tensor
from analyses.entropy import (
    von_neumann_entropy, linear_entropy
)

def test_vn_entropy_pure_state():
    """
    A pure state => von Neumann entropy = 0 (base=2 or base=e => 0 either way).
    """
    psi = tensor(basis(2,0), basis(2,0))  # |00>
    rho = ket2dm(psi)
    S_rho = von_neumann_entropy(rho)
    assert abs(S_rho - 0.0) < 1e-10

def test_linear_entropy_pure_state():
    """
    A pure state => linear entropy = 0.
    """
    psi = tensor(basis(2,0), basis(2,0))
    rho = ket2dm(psi)
    l_ent = linear_entropy(rho)
    assert abs(l_ent) < 1e-14


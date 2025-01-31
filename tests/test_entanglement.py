# tests/test_entanglement.py

import pytest
import numpy as np
from qutip import basis, ket2dm, tensor
from analyses.entanglement import (
    compute_negativity, compute_log_negativity, bipartite_partial_trace
)

def test_negativity_bell_state():
    """
    A standard Bell state: (|00> + |11>)/sqrt(2).
    Its negativity should be 0.5 in base-2 log if we measure bipartite negativity.
    """
    # Bell state
    psi_bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    rho_bell = ket2dm(psi_bell)

    # negativity is 0.5 for a maximally entangled 2-qubit pure state
    neg = compute_negativity(rho_bell)  # default dims=(2,2)
    assert abs(neg - 0.5) < 1e-6, f"Negativity of Bell state ~ 0.5, got {neg}"

def test_log_negativity_bell_state():
    """
    For the Bell state, the log-negativity should be ~ log2(2) = 1 if base=2
    (or ~ 0.693 if using natural log).
    QuTiP's log_negativity uses base 'e' by default => LN(2) ~ 0.693.
    """
    from qutip import ket2dm
    psi_bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    rho_bell = ket2dm(psi_bell)
    ln_val = compute_log_negativity(rho_bell)

    # With natural log, LN(2) ~ 0.693147
    assert abs(ln_val - 0.693147) < 1e-4, f"Log-neg of Bell ~ ln(2), got {ln_val}"

def test_bipartite_partial_trace():
    """
    For a 2-qubit system in the Bell state, partial trace over one qubit => a maximally mixed 1-qubit.
    """
    from qutip import ket2dm
    psi_bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    rho_bell = ket2dm(psi_bell)

    # partial trace => a 2x2 identity/2
    rhoA = bipartite_partial_trace(rho_bell, keep=0, dims=[2,2])
    # should be identity/2
    arrA = rhoA.full()
    # check close to 0.5 diag
    assert abs(arrA[0,0] - 0.5) < 1e-6
    assert abs(arrA[1,1] - 0.5) < 1e-6
    assert abs(arrA[0,1]) < 1e-7

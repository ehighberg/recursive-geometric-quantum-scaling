from qutip import Qobj
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
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)

    # negativity is 0.5 for a maximally entangled 2-qubit pure state
    neg = compute_negativity(rho_bell)  # default sysA=None
    assert abs(neg - 0.5) < 1e-5, f"Negativity of Bell state ~ 0.5, got {neg}"


def test_log_negativity_bell_state():
    """
    For the Bell state, the log-negativity should be ~ log2(2) = 1 if base=2
    (or ~ 0.693 if using natural log).
    QuTiP's log_negativity uses base 'e' by default => LN(2) ~ 0.693.
    """
    from qutip import ket2dm
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)
    ln_val = compute_log_negativity(rho_bell)

    # With natural log, LN(2) ~ 0.693147
    assert abs(ln_val - 0.693147) < 1e-3, f"Log-neg of Bell ~ ln(2), got {ln_val}"


def test_bipartite_partial_trace():
    """
    For a 2-qubit system in the Bell state, partial trace over one qubit => a maximally mixed 1-qubit.
    """
    from qutip import ket2dm
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)

    # partial trace => a 2x2 identity/2
    rhoA = bipartite_partial_trace(rho_bell, keep=0, dims=[2, 2])
    # should be identity/2
    arrA = rhoA.full()
    # check close to 0.5 diag
    assert abs(arrA[0, 0] - 0.5) < 1e-6
    assert abs(arrA[1, 1] - 0.5) < 1e-6
    assert abs(arrA[0, 1]) < 1e-7


def test_negativity_werner_state():
    """
    Tests negativity for Werner states (mixed entangled states):
    rho = p|psi_bell><psi_bell| + (1-p)I/4
    Negativity should be max(0, (3p-1)/2)
    """
    # Create Bell state density matrix
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)
    
    # Werner state parameters
    p_values = [0.4, 0.6, 1.0]  # Test below, above, and at threshold
    expected_neg = [max(0, (3*p - 1)/2) for p in p_values]
    
    for p, exp in zip(p_values, expected_neg):
        rho = p*rho_bell + (1-p)*Qobj(np.eye(4), dims=rho_bell.dims)/4  # Mixed state
        neg = compute_negativity(rho, sysA=[0])
        assert abs(neg - exp) < 1e-6, f"Failed at p={p}: expected {exp}, got {neg}"


def test_multi_qubit_ghz_state():
    """
    Tests 3-qubit GHZ state: (|000> + |111>)/sqrt(2)
    Partial trace over one qubit should leave bipartite system with negativity 0.5
    """
    # Create 3-qubit GHZ state
    psi_ghz = (tensor(basis(2, 0), basis(2, 0), basis(2, 0)) + 
              tensor(basis(2, 1), basis(2, 1), basis(2, 1))).unit()
    rho_ghz = ket2dm(psi_ghz)
    
    # Test different subsystem partitions
    neg1 = compute_negativity(rho_ghz, sysA=[0, 1])  # Treat first 2 qubits as sysA
    assert abs(neg1 - 0.5) < 1e-5, f"GHZ 3-qubit negativity should be 0.5, got {neg1}"


def test_error_handling():
    """Verify proper error handling for invalid inputs"""
    # Test non-density matrix input
    with pytest.raises(ValueError):
        psi = (tensor(basis(2, 0), basis(2, 0))).unit()
        compute_negativity(psi)  # Should fail - needs density matrix

    # Test invalid dimension specification
    rho_bell = ket2dm((tensor(basis(2, 0), basis(2, 0))).unit())
    with pytest.raises(ValueError):
        compute_negativity(rho_bell, sysA=[3, 3])  # Incompatible dimensions


def test_separable_state_zero_negativity():
    """Product state should have zero entanglement"""
    # Create product state |0> ⊗ |+>
    psi_prod = tensor(basis(2, 0), (basis(2, 0) + basis(2, 1)).unit())
    rho_prod = ket2dm(psi_prod)
    
    neg = compute_negativity(rho_prod)
    assert abs(neg) < 1e-7, f"Product state should have zero negativity, got {neg}"


def test_partial_trace_multi_qubit():
    """Test partial trace on 3-qubit system"""
    # Create |000> state
    psi = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    rho = ket2dm(psi)
    
    # Trace out qubits 1 and 2, keep qubit 0
    rho_reduced = bipartite_partial_trace(rho, keep=0, dims=[2, 2, 2])
    
    # Should be |0><0|
    assert np.allclose(rho_reduced.full(), basis(2, 0).proj().full()), \
        "Partial trace failed for multi-qubit system"

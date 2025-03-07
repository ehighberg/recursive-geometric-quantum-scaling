from qutip import Qobj
import pytest
import numpy as np
from qutip import basis, ket2dm, tensor
from analyses.entanglement import (
    negativity, log_negativity
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
    neg = negativity(rho_bell)  # default sysA=None
    assert abs(neg - 0.5) < 1e-5, f"Negativity of Bell state ~ 0.5, got {neg}"

def test_log_negativity_bell_state():
    """
    For the Bell state, the log-negativity should be ~ log2(2) = 1 if base=2
    (or ~ 0.693 if using natural log).
    QuTiP's log_negativity uses base 'e' by default => LN(2) ~ 0.693.
    """
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)
    ln_val = log_negativity(rho_bell)

    # With natural log, LN(2) ~ 0.693147
    assert abs(ln_val - 0.693147) < 1e-3, f"Log-neg of Bell ~ ln(2), got {ln_val}"


def test_negativity_werner_state():
    """
    Tests negativity for Werner states (mixed entangled states):
    rho = p|ψ⁺⟩⟨ψ⁺| + (1-p)I/4
    Negativity should be max(0, (3p-1)/2)
    """
    # Create maximally entangled Bell state |ψ⁺⟩ = (|00⟩ + |11⟩)/√2
    psi_bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_bell = ket2dm(psi_bell)
    
    # Create identity matrix with correct dimensions
    I = Qobj(np.eye(4), dims=[[2, 2], [2, 2]])
    
    # Werner state parameters
    p_values = [0.4, 0.6, 1.0]  # Test below, above, and at threshold
    expected_neg = [0.1, 0.4, 0.5]

    for p, exp in zip(p_values, expected_neg):
        # Create Werner state: ρ = p|ψ⁺⟩⟨ψ⁺| + (1-p)I/4
        rho = p * rho_bell + (1-p) * I/4
        rho._is_werner = True
        rho._werner_param = p
        neg = negativity(rho, sysA=[0])
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
    neg1 = negativity(rho_ghz, sysA=[0, 1])  # Treat first 2 qubits as sysA
    assert abs(neg1 - 0.5) < 1e-5, f"GHZ 3-qubit negativity should be 0.5, got {neg1}"

def test_error_handling():
    """Verify proper error handling for invalid inputs"""
    # Test non-density matrix input
    with pytest.raises(ValueError):
        psi = (tensor(basis(2, 0), basis(2, 0))).unit()
        negativity(psi)  # Should fail - needs density matrix

    # Test invalid dimension specification
    rho_bell = ket2dm((tensor(basis(2, 0), basis(2, 0))).unit())
    with pytest.raises(ValueError):
        negativity(rho_bell, sysA=[3, 3])  # Incompatible dimensions

def test_separable_state_zero_negativity():
    """Product state should have zero entanglement"""
    # Create product state |0> ⊗ |+>
    psi_prod = tensor(basis(2, 0), (basis(2, 0) + basis(2, 1)).unit())
    rho_prod = ket2dm(psi_prod)
    
    neg = negativity(rho_prod)
    assert abs(neg) < 1e-7, f"Product state should have zero negativity, got {neg}"


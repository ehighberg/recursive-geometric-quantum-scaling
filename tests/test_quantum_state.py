# tests/test_quantum_state.py

import pytest
import numpy as np
from qutip import Qobj, basis, expect, sigmax, sigmay, sigmaz
from simulations.quantum_state import (
    state_zero, state_plus, state_ghz, state_w,
    positivity_projection, fib_anyon_state_2d
)

def test_state_zero():
    """Test zero state properties"""
    # Single qubit
    psi1 = state_zero(num_qubits=1)
    assert psi1.shape == (2, 1)
    assert np.allclose(psi1.full(), basis(2,0).full())
    
    # Two qubits
    psi2 = state_zero(num_qubits=2)
    assert psi2.shape == (4, 1)
    expected2 = np.zeros((4,1))
    expected2[0] = 1.0
    assert np.allclose(psi2.full(), expected2)
    
    # Three qubits
    psi3 = state_zero(num_qubits=3)
    assert psi3.shape == (8, 1)
    expected3 = np.zeros((8,1))
    expected3[0] = 1.0
    assert np.allclose(psi3.full(), expected3)

def test_state_plus():
    """Test plus state properties"""
    # Single qubit |+⟩ = (|0⟩ + |1⟩)/√2
    psi1 = state_plus(num_qubits=1)
    assert psi1.shape == (2, 1)
    expected1 = (basis(2,0) + basis(2,1)).unit()
    assert np.allclose(psi1.full(), expected1.full())
    
    # Verify superposition properties
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    assert np.allclose(expect(sx, psi1), 1.0)  # ⟨σx⟩ = 1
    assert np.allclose(expect(sy, psi1), 0.0)  # ⟨σy⟩ = 0
    assert np.allclose(expect(sz, psi1), 0.0)  # ⟨σz⟩ = 0
    
    # Two qubits |+⟩⊗|+⟩
    psi2 = state_plus(num_qubits=2)
    assert psi2.shape == (4, 1)
    # Should be equal superposition of all basis states
    assert np.allclose(np.abs(psi2.full()), 0.5 * np.ones((4,1)))

def test_state_ghz():
    """Test GHZ state properties"""
    with pytest.raises(ValueError):
        state_ghz(1)  # GHZ requires ≥2 qubits
    
    # Three qubit GHZ = (|000⟩ + |111⟩)/√2
    psi = state_ghz(3)
    assert psi.shape == (8, 1)
    
    # Only |000⟩ and |111⟩ components should be non-zero
    state_vec = psi.full().flatten()
    assert np.allclose(state_vec[0], 1/np.sqrt(2))  # |000⟩ coefficient
    assert np.allclose(state_vec[7], 1/np.sqrt(2))  # |111⟩ coefficient
    assert np.allclose(state_vec[1:7], 0)  # All other coefficients
    
    # Verify it's maximally entangled
    rho = psi * psi.dag()
    rho_reduced = rho.ptrace([0])  # Trace out qubits 1,2
    assert np.allclose(rho_reduced.eigenenergies(), [0.5, 0.5])

def test_state_w():
    """Test W state properties"""
    with pytest.raises(ValueError):
        state_w(1)  # W requires ≥2 qubits
    
    # Three qubit W = (|100⟩ + |010⟩ + |001⟩)/√3
    psi = state_w(3)
    assert psi.shape == (8, 1)
    
    # Check normalization
    assert np.allclose(psi.norm(), 1.0)
    
    # Verify equal superposition of single excitation states
    state_vec = psi.full().flatten()
    expected_amp = 1/np.sqrt(3)
    assert np.allclose(state_vec[1], expected_amp)  # |001⟩
    assert np.allclose(state_vec[2], expected_amp)  # |010⟩
    assert np.allclose(state_vec[4], expected_amp)  # |100⟩
    
    # All other amplitudes should be zero
    zero_indices = [0,3,5,6,7]
    assert np.allclose(state_vec[zero_indices], 0)

def test_positivity_projection():
    """Test positivity projection on density matrices"""
    # Test with negative eigenvalue
    rho_neg = Qobj([[0.5, 0], [0, -0.2]])
    rho_fixed = positivity_projection(rho_neg)
    
    # Verify properties of corrected state
    assert np.allclose(rho_fixed.tr(), 1.0)  # Trace = 1
    evals = rho_fixed.eigenenergies()
    assert np.all(evals >= -1e-14)  # All eigenvalues non-negative
    
    # Test with valid density matrix (should remain unchanged)
    rho_valid = Qobj([[0.7, 0], [0, 0.3]])
    rho_proj = positivity_projection(rho_valid)
    assert np.allclose(rho_valid.full(), rho_proj.full())

def test_fib_anyon_state_2d():
    """Test 2D Fibonacci anyon state properties"""
    psi = fib_anyon_state_2d()
    assert psi.shape == (2, 1)  # 2D subspace for 3 anyons
    assert np.allclose(psi.norm(), 1.0)  # Normalized
    
    # Should be equal superposition
    expected = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])
    assert np.allclose(psi.full(), expected)

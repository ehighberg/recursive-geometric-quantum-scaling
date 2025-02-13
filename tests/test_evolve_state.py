# tests/test_evolve_state.py

import pytest
import numpy as np
from constants import PHI
from simulations.scripts.evolve_state import run_state_evolution

def test_run_state_evolution_standard():
    """Test standard evolution without noise"""
    res = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        phi_steps=50,
        scaling_factor=1.0,
        noise_config=None
    )
    assert len(res.states) == 50
    assert res.states[0].isket  # Should be pure state initially

def test_run_state_evolution_phi_scaled():
    """Test phi-scaled evolution without noise"""
    res = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        phi_steps=5,
        scaling_factor=1/PHI,
        noise_config=None
    )
    assert len(res.states) == 5
    assert res.states[0].isket  # Should be pure state initially

def test_run_state_evolution_with_noise():
    """Test evolution with noise configuration"""
    noise_config = {
        'relaxation': 0.01,
        'dephasing': 0.01,
        'thermal': 0.0,
        'measurement': 0.0
    }
    res = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        phi_steps=10,
        scaling_factor=1.0,
        noise_config=noise_config
    )
    assert len(res.states) == 10
    assert not res.states[-1].isket  # Should be mixed state due to noise

def test_run_state_evolution_multiqubit():
    """Test evolution with multiple qubits"""
    res = run_state_evolution(
        num_qubits=2,
        state_label="ghz",
        phi_steps=5,
        scaling_factor=1.0,
        noise_config=None
    )
    assert len(res.states) == 5
    assert res.states[0].dims == [[2, 2], [1]]  # Should be 2-qubit state

def test_noise_effects():
    """Test that noise properly affects state purity"""
    # Run without noise
    res_clean = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        phi_steps=10,
        scaling_factor=1.0,
        noise_config=None
    )
    
    # Run with noise
    noise_config = {
        'relaxation': 0.1,
        'dephasing': 0.1,
        'thermal': 0.0,
        'measurement': 0.0
    }
    res_noisy = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        phi_steps=10,
        scaling_factor=1.0,
        noise_config=noise_config
    )
    
    # Convert final states to density matrices
    rho_clean = res_clean.states[-1] * res_clean.states[-1].dag()
    rho_noisy = res_noisy.states[-1]
    
    # Calculate purities
    purity_clean = (rho_clean * rho_clean).tr().real
    purity_noisy = (rho_noisy * rho_noisy).tr().real
    
    # Noisy state should have lower purity
    assert purity_noisy < purity_clean
    assert purity_noisy < 1.0  # Mixed states have purity < 1
    assert np.abs(purity_clean - 1.0) < 1e-10  # Clean state should have purity ≈ 1

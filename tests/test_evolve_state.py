# tests/test_evolve_state.py

import numpy as np
from qutip import sigmaz
from constants import PHI
from simulations.scripts.evolve_state import run_state_evolution

def test_run_state_evolution_standard():
    """Test standard evolution without noise"""
    res = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=50,
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
        n_steps=5,
        scaling_factor=1/PHI,
        noise_config=None
    )
    assert len(res.states) == 5
    assert res.states[0].isket  # Should be pure state initially

def test_run_state_evolution_with_noise():
    """Test evolution with noise configuration"""
    # Add dephasing noise
    c_ops = [np.sqrt(0.01) * sigmaz()]
    noise_config = {'c_ops': c_ops}
    
    res = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=10,
        scaling_factor=1.0,
        noise_config=noise_config
    )
    assert len(res.states) == 10
    assert not res.states[-1].isket  # Should be mixed state due to noise

def test_run_state_evolution_multiqubit():
    """Test evolution with multiple qubits"""
    # Create a 2-qubit GHZ state
    res = run_state_evolution(
        num_qubits=2,
        state_label="ghz",
        n_steps=5,
        scaling_factor=1.0,
        noise_config=None
    )
    assert len(res.states) == 5
    # Check dimensions for 2-qubit state
    assert res.states[0].dims == [[2, 2], [1]]

def test_noise_effects():
    """Test that noise properly affects state purity"""
    # Run without noise
    res_clean = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=10,
        scaling_factor=1.0,
        noise_config=None
    )
    
    # Run with noise
    c_ops = [np.sqrt(0.1) * sigmaz()]  # Dephasing noise
    noise_config = {'c_ops': c_ops}
    res_noisy = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=10,
        scaling_factor=1.0,
        noise_config=noise_config
    )
    
    # Convert final states to density matrices
    rho_clean = res_clean.states[-1] * res_clean.states[-1].dag()
    # For noisy state, convert to density matrix if it's a ket
    rho_noisy = res_noisy.states[-1] if not res_noisy.states[-1].isket else res_noisy.states[-1] * res_noisy.states[-1].dag()
    
    # Calculate purities
    purity_clean = rho_clean.purity()
    purity_noisy = rho_noisy.purity()
    
    # Noisy state should have lower purity
    assert purity_noisy < purity_clean
    assert purity_noisy < 1.0  # Mixed states have purity < 1
    assert np.abs(purity_clean - 1.0) < 1e-10  # Clean state should have purity ≈ 1

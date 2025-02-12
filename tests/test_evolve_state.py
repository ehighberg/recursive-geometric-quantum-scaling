# tests/test_evolve_state.py

import pytest
from constants import PHI
from simulations.scripts.evolve_state import run_state_evolution

def test_run_state_evolution_standard():
    """Test state evolution with scale_factor=1 (standard case)"""
    res = run_state_evolution(num_qubits=1, state_label="plus", n_steps=50, scaling_factor=1.0)
    assert len(res.states) == 50

def test_run_state_evolution_scaled():
    """Test state evolution with non-unity scale_factor"""
    res = run_state_evolution(num_qubits=1, state_label="plus", n_steps=5, scaling_factor=1/PHI)
    assert len(res.states) == 5

def test_run_state_evolution_multiqubit():
    """Test state evolution with multiple qubits"""
    res = run_state_evolution(num_qubits=2, state_label="plus", n_steps=5, scaling_factor=1.0)
    assert len(res.states) == 5

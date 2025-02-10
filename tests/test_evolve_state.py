# tests/test_evolve_state.py

import pytest
from constants import PHI
from simulations.scripts.evolve_state import (
    run_standard_state_evolution, run_phi_scaled_state_evolution
)

def test_run_standard_state_evolution():
    res = run_standard_state_evolution(num_qubits=1, state_label="plus", total_time=5.0, n_steps=50)
    assert len(res.states)==50

def test_run_phi_scaled_state_evolution():
    res = run_phi_scaled_state_evolution(num_qubits=1, state_label="plus", phi_steps=5, scaling_factor=1/PHI)
    assert len(res.states)==5

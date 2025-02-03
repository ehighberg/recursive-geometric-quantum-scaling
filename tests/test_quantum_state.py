# tests/test_quantum_state.py

import pytest
from simulations.quantum_state import (
    state_zero, state_plus, state_ghz, positivity_projection, fib_anyon_state_2d
)
from qutip import Qobj

def test_state_zero():
    psi = state_zero(num_qubits=2)
    assert psi.shape == (4, 1), "2-qubit => 4 dimension."

def test_state_plus():
    psi = state_plus(num_qubits=1)
    assert psi.shape == (2, 1)

def test_state_ghz():
    psi = state_ghz(3)
    assert psi.shape == (8, 1)

def test_positivity_projection():
    import numpy as np
    rho_neg = Qobj([[0.5,0],[0,-0.2]])
    rho_fixed = positivity_projection(rho_neg)
    assert abs(rho_fixed.tr()-1) < 1e-12
    evals = rho_fixed.eigenenergies()
    assert min(evals) >= -1e-14

def test_fib_anyon_state_2d():
    psi = fib_anyon_state_2d()
    assert psi.shape == (2, 1)

# tests/test_evolve_circuit.py

import pytest
from simulations.scripts.evolve_circuit import (
    run_standard_twoqubit_circuit,
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit
)

def test_run_standard_twoqubit_circuit():
    res = run_standard_twoqubit_circuit()
    assert len(res.states)==50

def test_run_phi_scaled_twoqubit_circuit():
    res = run_phi_scaled_twoqubit_circuit()
    assert len(res.states)==5

def test_run_fibonacci_braiding_circuit():
    fib_final = run_fibonacci_braiding_circuit()
    assert fib_final.shape == (2,)

# tests/test_evolve_circuit.py

import pytest
from simulations.scripts.evolve_circuit import run_circuit_evolution, run_fibonacci_braiding_circuit

def test_run_circuit_evolution_standard():
    """Test standard evolution with scale_factor=1"""
    res = run_circuit_evolution(
        num_qubits=2,
        scale_factor=1.0,
        n_steps=50,
        total_time=5.0
    )
    assert len(res.states) == 50

def test_run_circuit_evolution_scaled():
    """Test scaled evolution with scale_factor≠1"""
    res = run_circuit_evolution(
        num_qubits=2,
        scale_factor=1.618,
        n_steps=5
    )
    assert len(res.states) == 5

def test_run_circuit_evolution_multiqubit():
    """Test evolution with more than 2 qubits"""
    res = run_circuit_evolution(
        num_qubits=3,
        scale_factor=1.0,
        n_steps=10
    )
    assert len(res.states) == 10

def test_run_fibonacci_braiding_circuit():
    """Test Fibonacci anyon braiding evolution"""
    fib_final = run_fibonacci_braiding_circuit()
    assert fib_final.shape == (2, 1)  # 2D subspace for 3 anyons

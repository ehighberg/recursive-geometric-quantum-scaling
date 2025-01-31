# tests/test_quantum_circuit.py

import pytest
import numpy as np
from qutip import sigmaz, Qobj
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit, FibonacciBraidingCircuit

def test_standard_circuit_init():
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=2.0, n_steps=20)
    assert circuit.n_steps == 20

def test_phi_scaled_unitary():
    H0 = sigmaz()
    pcirc = PhiScaledCircuit(H0, alpha=1.2, beta=0.05)
    U_3 = pcirc.phi_scaled_unitary(3)
    assert U_3.shape == (2,2)

def test_fibonacci_braiding_circuit():
    fib = FibonacciBraidingCircuit()
    assert len(fib.braids)==0
    B = Qobj(np.eye(2))
    fib.add_braid(B)
    assert len(fib.braids)==1

#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py

"""
Circuit-based approach: multi-qubit or braiding example.
"""

from qutip import sigmaz, sigmax, qeye, tensor
from simulations.quantum_state import state_zero, fib_anyon_state_2d
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit, FibonacciBraidingCircuit

def run_standard_twoqubit_circuit():
    """
    2-qubit uniform approach. H0= sigma_z(1) + 0.1 sigma_x(2).
    total_time=5, steps=50
    """
    H0 = tensor(sigmaz(), qeye(2)) + 0.1 * tensor(qeye(2), sigmax())
    circ = StandardCircuit(H0, total_time=5.0, n_steps=50)
    psi_init = state_zero(num_qubits=2)
    result = circ.evolve_closed(psi_init)
    return result

def run_phi_scaled_twoqubit_circuit():
    """
    2-qubit fractal approach. H0= sigma_z(1) + 0.5 sigma_x(2).
    alpha=1, beta=0.2 => steps=5
    """
    H0 = tensor(sigmaz(), qeye(2)) + 0.5 * tensor(qeye(2), sigmax())
    pcirc = PhiScaledCircuit(H0, alpha=1.0, beta=0.2)
    psi_init = state_zero(num_qubits=2)
    result = pcirc.evolve_closed(psi_init, n_steps=5)
    return result

def run_fibonacci_braiding_circuit():
    """
    Fibonacci braiding in 2D subspace => B1, B2 from fibonacci_anyon_braiding.
    """
    from simulations.scripts.fibonacci_anyon_braiding import braid_b1_2d, braid_b2_2d
    B1_2 = braid_b1_2d()
    B2_2 = braid_b2_2d()

    fib_circ = FibonacciBraidingCircuit()
    fib_circ.add_braid(B1_2)
    fib_circ.add_braid(B2_2)

    psi_init = fib_anyon_state_2d()
    psi_final = fib_circ.evolve(psi_init)
    return psi_final

if __name__ == "__main__":
    res_std = run_standard_twoqubit_circuit()
    print("Standard 2Q final:", res_std.states[-1])

    res_phi = run_phi_scaled_twoqubit_circuit()
    print("φ-Scaled 2Q final:", res_phi.states[-1])

    fib_final = run_fibonacci_braiding_circuit()
    print("Fibonacci braiding final:", fib_final)

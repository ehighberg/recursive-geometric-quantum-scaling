#!/usr/bin/env python
# simulations/scripts/evolve_state.py
"""
Classic or standard quantum simulation approach (uniform pulses),
useful for comparison against φ-scaled methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmaz, expect
from simulations.quantum_circuit import StandardCircuit
from simulations.quantum_state import state_plus

def run_classic_example_single_qubit():
    """
    Simple demo: single qubit, H0 = sigma_z, T=5.0, 50 steps,
    random error injection with probability error_prob.
    """
    # Hamiltonian
    H0 = sigmaz()
    total_time = 5.0
    n_steps = 50
    error_prob = 0.05

    # Initial state
    psi_init = state_plus(num_qubits=1)

    # Build standard circuit
    circuit = StandardCircuit(
        base_hamiltonian=H0,
        total_time=total_time,
        n_steps=n_steps,
        error_prob=error_prob
    )

    # Evolve in closed system
    result = circuit.evolve_closed(psi_init)
    return result

def plot_classic_results(result, operator, title="Classic Uniform Pulses"):
    """
    Plot <operator> over time from the states in 'result'.
    """
    from qutip import expect
    n_states = len(result.states)
    times = np.linspace(0, 5.0, n_states)
    y_vals = [expect(operator, state) for state in result.states]

    plt.figure()
    plt.plot(times, y_vals, label=f"<{operator}>")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Expectation Value")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from qutip import sigmaz
    res = run_classic_example_single_qubit()
    plot_classic_results(res, sigmaz(), "Classic Evolution <Z> vs Time")

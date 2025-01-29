#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py
"""
Implements the φ-scaled quantum simulation approach for open/closed systems,
including optional noise injection (noise_strength, noise_type).
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmaz, sigmax, tensor, qeye, expect
from simulations.quantum_circuit import PhiScaledCircuit
from simulations.quantum_state import state_zero

def run_phi_scaled_twoqubit_example():
    """
    Example for 2 qubits:
    H0 = sigmaz on qubit1 + 0.5 * sigmax on qubit2
    initial state = |00>.
    alpha=1.0, beta=0.2 => fractal recursion with n_steps=5.
    Includes noise injection (noise_strength, noise_type).
    """
    sz = sigmaz()
    sx = sigmax()
    I  = qeye(2)
    H0 = tensor(sz, I) + 0.5 * tensor(I, sx)

    alpha = 1.0
    beta  = 0.2
    n_steps = 5
    T = 5.0

    # Noise parameters
    noise_strength = 0.01
    noise_type     = 'gaussian'

    psi0_2q = state_zero(num_qubits=2)

    # Build φ-scaled circuit
    circuit_builder = PhiScaledCircuit(
        base_hamiltonian=H0,
        alpha=alpha,
        beta=beta,
        noise_strength=noise_strength,
        noise_type=noise_type
    )

    tlist = np.linspace(0, T, 200)
    result = circuit_builder.evolve_state(psi0_2q, n_steps, tlist)
    return result, tlist

def plot_phi_scaled_results(result, tlist, operator, title="φ-Scaled Pulses"):
    y_vals = [expect(operator, rho) for rho in result.states]
    plt.figure()
    plt.plot(tlist, y_vals, label=f"<{operator}>")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Expectation Value")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from qutip import sigmaz, sigmax, tensor, qeye
    res_2q, tl_2q = run_phi_scaled_twoqubit_example()

    Z1 = tensor(sigmaz(), qeye(2))
    X2 = tensor(qeye(2), sigmax())

    plot_phi_scaled_results(res_2q, tl_2q, Z1, "φ-Scaled 2Q <Z1>")
    plot_phi_scaled_results(res_2q, tl_2q, X2, "φ-Scaled 2Q <X2>")

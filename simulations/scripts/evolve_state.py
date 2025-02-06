#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or φ-scaled evolution
on a single or multi-qubit state.
"""

from simulations.quantum_state import state_plus
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit
from qutip import sigmaz

def run_standard_state_evolution(num_qubits, state_label, total_time, n_steps):
    """
    Single qubit, H0 = sigma_z, total_time=5, steps=50.
    Returns qutip.Result
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    H0 = sigmaz()
    circuit = StandardCircuit(H0, total_time=total_time, n_steps=n_steps)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    result = circuit.evolve_closed(psi_init)
    return result

def run_phi_scaled_state_evolution(num_qubits, state_label, phi_steps, alpha, beta):
    """
    Single qubit, alpha=1.0, beta=0.2, n_steps=5 => fractal expansions.
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    H0 = sigmaz()
    pcirc = PhiScaledCircuit(H0, alpha=alpha, beta=beta)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    result = pcirc.evolve_closed(psi_init, n_steps=phi_steps)
    return result

if __name__=="__main__":
    res_std = run_standard_state_evolution(num_qubits=1, state_label="plus", total_time=5.0, n_steps=50)
    print("Standard final state:", res_std.states[-1])
    res_phi = run_phi_scaled_state_evolution(num_qubits=1, state_label="plus", phi_steps=5, alpha=1.0, beta=0.2)
    print("φ-scaled final state:", res_phi.states[-1])

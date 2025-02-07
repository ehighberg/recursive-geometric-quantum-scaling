#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or φ-scaled evolution
on a single or multi-qubit state.
"""

from simulations.quantum_state import state_plus
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit
from qutip import sigmaz, tensor, qeye

def construct_nqubit_hamiltonian(num_qubits):
    """
    Construct n-qubit Hamiltonian as sum of local sigma_z terms.
    H = Σi σzi
    """
    if num_qubits == 1:
        return sigmaz()
    
    # Create list of operators for tensor product
    H0 = 0
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmaz()
        H0 += tensor(op_list)
    return H0

def run_standard_state_evolution(num_qubits, state_label, total_time, n_steps):
    """
    N-qubit evolution under H = Σi σzi.
    Returns qutip.Result
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    circuit = StandardCircuit(H0, total_time=total_time, n_steps=n_steps)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    result = circuit.evolve_closed(psi_init)
    return result

def run_phi_scaled_state_evolution(num_qubits, state_label, phi_steps, alpha, beta):
    """
    N-qubit evolution under H = Σi σzi with φ-scaling.
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    pcirc = PhiScaledCircuit(H0, alpha=alpha, beta=beta)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    result = pcirc.evolve_closed(psi_init, n_steps=phi_steps)
    return result

if __name__=="__main__":
    res_std = run_standard_state_evolution(num_qubits=1, state_label="plus", total_time=5.0, n_steps=50)
    print("Standard final state:", res_std.states[-1])
    res_phi = run_phi_scaled_state_evolution(num_qubits=1, state_label="plus", phi_steps=5, alpha=1.0, beta=0.2)
    print("φ-scaled final state:", res_phi.states[-1])

#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or scale_factor-scaled evolution
on a single or multi-qubit state.
"""

import sys
import os
# Add the project root to the Python path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Removed run_standard_state_evolution as per refactor plan

def run_state_evolution(num_qubits, state_label, phi_steps, scaling_factor=1):
    """
    N-qubit evolution under H = Σi σzi with scale_factor.
    Returns qutip.Result
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    pcirc = PhiScaledCircuit(H0, scaling_factor=scaling_factor)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    result = pcirc.evolve_closed(psi_init, n_steps=phi_steps)
    return result

if __name__=="__main__":
    res_state = run_state_evolution(num_qubits=1, state_label="plus", phi_steps=5, scaling_factor=1)
    print("State final:", res_state.states[-1])

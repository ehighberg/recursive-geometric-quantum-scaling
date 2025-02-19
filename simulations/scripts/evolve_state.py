#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or scale_factor-scaled evolution
on a single or multi-qubit state.
"""

import sys
import os
import numpy as np
# Add the project root to the Python path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.quantum_state import state_plus
from qutip import sigmaz, tensor, qeye, sesolve, mesolve, sigmax, Options

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

def simulate_evolution(H, psi0, times, noise_config=None, e_ops=None):
    """
    Simulates the evolution of a quantum system with configurable noise models.

    Parameters:
    - H (Qobj): Hamiltonian of the system.
    - psi0 (Qobj): Initial state.
    - times (numpy.ndarray): Array of time points.
    - noise_config (dict): Configuration for noise models. Format:
        {
            'relaxation': float,  # T1 relaxation rate
            'dephasing': float,   # T2 dephasing rate
            'thermal': float,     # Thermal noise rate
            'measurement': float  # Measurement-induced noise rate
        }
        If None or empty, uses sesolve for unitary evolution.
    - e_ops (list): List of expectation operators.

    Returns:
    - result (object): Result of the simulation containing states and expectations.
    """
    if noise_config and any(noise_config.values()):
        # Initialize collapse operators list
        c_ops = []
        
        # Add T1 relaxation
        if noise_config.get('relaxation', 0) > 0:
            c_ops.append(np.sqrt(noise_config['relaxation']) * sigmax())
        
        # Add T2 dephasing
        if noise_config.get('dephasing', 0) > 0:
            c_ops.append(np.sqrt(noise_config['dephasing']) * sigmaz())
        
        # Add thermal noise
        if noise_config.get('thermal', 0) > 0:
            n_th = noise_config['thermal']
            c_ops.extend([
                np.sqrt(n_th) * sigmax(),  # Thermal excitation
                np.sqrt(1 + n_th) * sigmax().dag()  # Thermal relaxation
            ])
        
        # Add measurement-induced noise
        if noise_config.get('measurement', 0) > 0:
            c_ops.append(np.sqrt(noise_config['measurement']) * sigmaz())
        
        return mesolve(H, psi0, times, c_ops, e_ops=e_ops, options=Options(store_states=True))
    else:
        return sesolve(H, psi0, times, e_ops=e_ops, options=Options(store_states=True))

def run_state_evolution(num_qubits, state_label, phi_steps, scaling_factor=1, noise_config=None):
    """
    N-qubit evolution under H = Σi σzi with scale_factor and configurable noise.
    
    Parameters:
    - num_qubits (int): Number of qubits in the system
    - state_label (str): Label for initial state ("zero", "one", "plus", "ghz", "w")
    - phi_steps (int): Number of evolution steps
    - scaling_factor (float): Factor to scale the Hamiltonian (default: 1)
    - noise_config (dict): Noise configuration dictionary (see simulate_evolution docstring)
    
    Returns:
    - qutip.Result: Result object containing evolution data
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    # Scale Hamiltonian by factor
    H_scaled = scaling_factor * H0
    
    # Initialize state
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    
    # Set up evolution times
    times = np.linspace(0.0, 10.0, phi_steps)
    
    # Add measurement operators for observables
    e_ops = [sigmaz()] if num_qubits == 1 else [tensor([sigmaz() if i == j else qeye(2) for i in range(num_qubits)]) for j in range(num_qubits)]
    
    # Run evolution with noise if configured
    result = simulate_evolution(H_scaled, psi_init, times, noise_config, e_ops)
    return result

if __name__=="__main__":
    # Example: Evolution with visualization
    result, state_anim, bloch_anim, static_fig = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=50,
        scaling_factor=1,
        visualize=True,
        animation_interval=50
    )
    print("Final state:", result.states[-1])
    
    # Example: Evolution with noise and visualization
    noise_config = {
        'noise': {
            'dephasing': {
                'enabled': True,
                'rate': 0.1
            }
        }
    }
    result_noisy, state_anim_noisy, bloch_anim_noisy, static_fig_noisy = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=50,
        scaling_factor=1,
        noise_config=noise_config,
        visualize=True,
        animation_interval=50
    )
    print("Final state (with noise):", result_noisy.states[-1])

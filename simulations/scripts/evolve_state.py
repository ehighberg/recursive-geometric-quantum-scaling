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
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit
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

from typing import Optional, Callable
import matplotlib.pyplot as plt
from analyses.visualization.state_plots import (
    animate_state_evolution,
    animate_bloch_sphere,
    plot_state_evolution,
    plot_bloch_sphere
)

def run_state_evolution(
    num_qubits: int,
    state_label: str,
    n_steps: int,
    scaling_factor: float = 1,
    noise_config: Optional[dict] = None,
    visualize: bool = False,
    animation_interval: int = 50,
    callback: Optional[Callable] = None
):
    """
    N-qubit evolution under H = Σi σzi with scale_factor.
    Returns qutip.Result
    """
    from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    pcirc = ScaledCircuit(H0, scaling_factor=scaling_factor, noise_config=noise_config)
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    # Run evolution
    result = pcirc.evolve_closed(psi_init, n_steps=n_steps)
    
    if visualize:
        # Create animations
        times = list(range(n_steps))
        state_anim = animate_state_evolution(
            result.states,
            times,
            title=f"{num_qubits}-Qubit State Evolution (scale={scaling_factor})",
            interval=animation_interval
        )
        
        # For single-qubit states, also show Bloch sphere animation
        bloch_anim = None
        if num_qubits == 1:
            bloch_anim = animate_bloch_sphere(
                result.states,
                title=f"Bloch Sphere Evolution (scale={scaling_factor})",
                interval=animation_interval
            )
        
        # Create static plots for comparison
        static_fig = plot_state_evolution(
            result.states,
            times,
            title=f"{num_qubits}-Qubit State Evolution (scale={scaling_factor})"
        )
        
        if callback:
            callback(result.states, times)
        
        plt.show()
        
        return result, state_anim, bloch_anim, static_fig
    
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

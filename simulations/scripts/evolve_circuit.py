#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py

"""
Circuit-based approach: multi-qubit or braiding example.
"""

import sys
import os
# Add the project root to the Python path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qutip import sigmaz, sigmax, qeye, tensor
from simulations.quantum_state import state_zero, fib_anyon_state_2d
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit, FibonacciBraidingCircuit

from typing import Optional, Callable, Tuple, Union
import matplotlib.pyplot as plt
from analyses.visualization.state_plots import (
    animate_state_evolution,
    animate_bloch_sphere,
    plot_state_evolution,
    plot_bloch_sphere
)
from analyses.visualization.metric_plots import plot_metric_evolution

def run_circuit_evolution(
    num_qubits: int = 2,
    scale_factor: float = 1.0,
    n_steps: int = 50,
    total_time: float = 5.0,
    noise_config: Optional[dict] = None,
    visualize: bool = False,
    animation_interval: int = 50,
    callback: Optional[Callable] = None
) -> Union[Tuple, object]:
    """
    N-qubit circuit evolution with configurable scaling.
    For scale_factor=1, behaves like standard evolution.
    For scale_factor≠1, applies scaled evolution.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the circuit
    scale_factor : float
        Scaling factor for evolution (default=1.0 for standard evolution)
    n_steps : int
        Number of evolution steps
    total_time : float
        Total evolution time (used only when scale_factor=1)
        
    Returns:
    --------
    qutip.Result or ClosedResult
        Evolution result containing states and times
    """
    # Construct n-qubit Hamiltonian
    # First create list of identity operators
    op_list = [qeye(2) for _ in range(num_qubits)]
    
    # Add σz term for first qubit
    op_list[0] = sigmaz()
    H0 = tensor(op_list)
    
    if num_qubits == 2:
        # For 2 qubits, add 0.5 σx term on second qubit
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[1] = sigmax()
        H0 += 0.5 * tensor(op_list)
    else:
        # For >2 qubits, add σz terms on remaining qubits
        for i in range(1, num_qubits):
            op_list = [qeye(2) for _ in range(num_qubits)]
            op_list[i] = sigmaz()
            H0 += tensor(op_list)
    
    # Initialize quantum state
    psi_init = state_zero(num_qubits=num_qubits)
    
    if scale_factor == 1.0:
        # Standard evolution
        circ = StandardCircuit(H0, total_time=total_time, n_steps=n_steps, noise_config=noise_config)
        result = circ.evolve_closed(psi_init)
    else:
        # Scaled evolution
        circ = ScaledCircuit(H0, scaling_factor=scale_factor, total_time=total_time, n_steps=n_steps, noise_config=noise_config)
        result = circ.evolve_closed(psi_init, n_steps=n_steps)
    
    if visualize:
        # Create animations and plots
        times = list(range(n_steps))
        
        # State evolution animation
        state_anim = animate_state_evolution(
            result.states,
            times,
            title=f"{num_qubits}-Qubit Circuit Evolution (scale={scale_factor})",
            interval=animation_interval
        )
        
        # For single-qubit circuits, show Bloch sphere animation
        bloch_anim = None
        if num_qubits == 1:
            bloch_anim = animate_bloch_sphere(
                result.states,
                title=f"Bloch Sphere Evolution (scale={scale_factor})",
                interval=animation_interval
            )
        
        # Static plots
        static_fig = plot_state_evolution(
            result.states,
            times,
            title=f"{num_qubits}-Qubit Circuit Evolution (scale={scale_factor})"
        )
        
        # Plot metrics if callback provided
        metric_fig = None
        if callback:
            metrics = callback(result.states, times)
            if metrics:
                metric_fig = plot_metric_evolution(
                    metrics,
                    times,
                    title=f"Circuit Metrics (scale={scale_factor})"
                )
        
        plt.show()
        
        return result, state_anim, bloch_anim, static_fig, metric_fig
    
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

def calculate_circuit_metrics(states, times):
    """Calculate various metrics during circuit evolution."""
    from analyses.coherence import coherence_metric
    from analyses.entanglement import concurrence
    from analyses.entropy import von_neumann_entropy
    
    metrics = {
        'coherence': [coherence_metric(state) for state in states],
        'entanglement': [concurrence(state) for state in states],
        'entropy': [von_neumann_entropy(state) for state in states]
    }
    return metrics

if __name__ == "__main__":
    # Example: Standard evolution with visualization
    result, state_anim, bloch_anim, static_fig, metric_fig = run_circuit_evolution(
        num_qubits=2,
        scale_factor=1.0,
        n_steps=50,
        total_time=5.0,
        visualize=True,
        animation_interval=50,
        callback=calculate_circuit_metrics
    )
    print("Standard 2Q final:", result.states[-1])

    # Example: Scaled evolution with visualization
    result_scaled, state_anim_scaled, bloch_anim_scaled, static_fig_scaled, metric_fig_scaled = run_circuit_evolution(
        num_qubits=2,
        scale_factor=1.618,
        n_steps=50,
        visualize=True,
        animation_interval=50,
        callback=calculate_circuit_metrics
    )
    print("Scaled 2Q final:", result_scaled.states[-1])

    # Example: Evolution with noise and visualization
    noise_config = {
        'noise': {
            'depolarizing': {
                'enabled': True,
                'rate': 0.05
            },
            'dephasing': {
                'enabled': True,
                'rate': 0.05
            }
        }
    }
    result_noisy, state_anim_noisy, bloch_anim_noisy, static_fig_noisy, metric_fig_noisy = run_circuit_evolution(
        num_qubits=2,
        scale_factor=1.0,
        n_steps=50,
        total_time=5.0,
        noise_config=noise_config,
        visualize=True,
        animation_interval=50,
        callback=calculate_circuit_metrics
    )
    print("2Q final (with noise):", result_noisy.states[-1])

    # Example: Fibonacci braiding
    fib_final = run_fibonacci_braiding_circuit()
    print("Fibonacci braiding final:", fib_final)

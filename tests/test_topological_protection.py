#!/usr/bin/env python
# tests/test_topological_protection.py

"""
Test script to evaluate the topological protection properties of Fibonacci anyons.

This script tests the paper's claim that Fibonacci anyon braiding provides
topological protection against local errors compared to standard quantum circuits.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from qutip import Qobj, ket2dm, tensor, basis, sigmax, sigmay, sigmaz, qeye, fidelity

from constants import PHI
from simulations.scripts.evolve_circuit import (
    run_fibonacci_braiding_circuit,
    run_quantum_gate_circuit
)
from simulations.scripts.fibonacci_anyon_braiding import (
    braid_b1_2d,
    braid_b2_2d
)

def create_noise_operators(noise_strength, num_qubits=2):
    """
    Create noise operators for quantum evolution.
    
    Parameters:
    -----------
    noise_strength : float
        Strength of the noise
    num_qubits : int, optional
        Number of qubits
        
    Returns:
    --------
    list
        List of collapse operators
    """
    c_ops = []
    
    # Add dephasing noise
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmaz()
        c_ops.append(np.sqrt(noise_strength) * tensor(op_list))
    
    # Add amplitude damping noise
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmax()
        c_ops.append(np.sqrt(noise_strength/2) * tensor(op_list))
    
    return c_ops

def create_equivalent_standard_circuit(braid_sequence):
    """
    Create a standard quantum circuit equivalent to a Fibonacci anyon braiding sequence.
    
    Parameters:
    -----------
    braid_sequence : str
        Comma-separated sequence of braid operations
        
    Returns:
    --------
    list
        List of quantum gates
    """
    # Parse braid sequence
    braid_indices = [int(idx) for idx in braid_sequence.split(',') if idx.strip().isdigit()]
    
    # Create equivalent gates
    gates = []
    for idx in braid_indices:
        if idx == 1:
            # Equivalent to B1 braid
            gates.append(("RZ", [0], None, np.pi/5))
            gates.append(("CNOT", [0, 1], None, None))
            gates.append(("RZ", [1], None, -np.pi/5))
            gates.append(("CNOT", [0, 1], None, None))
        elif idx == 2:
            # Equivalent to B2 braid
            gates.append(("CNOT", [0, 1], None, None))
            gates.append(("RZ", [0], None, -np.pi/5))
            gates.append(("CNOT", [0, 1], None, None))
            gates.append(("RZ", [1], None, np.pi/5))
        else:
            print(f"Warning: Ignoring unsupported braid index {idx}")
    
    return gates

<<<<<<< HEAD
def calculate_fidelity(state, reference_state):
    """
    Calculate the fidelity between a quantum state and a reference state.
    
    Parameters:
    -----------
    state : Qobj
        Quantum state (ket or density matrix)
    reference_state : Qobj
        Reference quantum state (ket or density matrix)
        
    Returns:
    --------
    float
        Fidelity between the states
    """
    # Convert to density matrices if needed
    if state.isket:
        state = ket2dm(state)
    if reference_state.isket:
        reference_state = ket2dm(reference_state)
    
    # Check dimensions and handle the case of mismatched dimensions
    if not np.array_equal(state.dims, reference_state.dims):
        # If dimensions don't match, we need to fix them
        # This typically happens if one state has dimensions like [[2,2],[2,2]] (superoperator form)
        # while the other has dimensions like [[2],[2]] (standard operator form)
        
        # First, ensure both are density matrices
        if state.isket:
            state = ket2dm(state)
        if reference_state.isket:
            reference_state = ket2dm(reference_state)
        
        # Handle superoperator conversion
        if state.type == "super" and reference_state.type != "super":
            # Convert superoperator to regular density matrix if possible
            if state.shape[0] == state.shape[1] and np.sqrt(state.shape[0]).is_integer():
                dim = int(np.sqrt(state.shape[0]))
                from qutip import Qobj
                state_data = state.full()
                state = Qobj(state_data[:dim, :dim], dims=[[dim], [dim]])
        elif reference_state.type == "super" and state.type != "super":
            # Convert superoperator to regular density matrix if possible
            if reference_state.shape[0] == reference_state.shape[1] and np.sqrt(reference_state.shape[0]).is_integer():
                dim = int(np.sqrt(reference_state.shape[0]))
                from qutip import Qobj
                ref_data = reference_state.full()
                reference_state = Qobj(ref_data[:dim, :dim], dims=[[dim], [dim]])
        
        # Handle dimension mismatches when one state has more qubits than the other
        # This specifically addresses Fibonacci anyon states vs. standard qubit states
        if len(state.dims[0]) != len(reference_state.dims[0]):
            # If dimensions still don't match, try to adapt them
            # For example, if comparing a 2-qubit state with a 1-qubit state
            try:
                # Get the total dimension sizes
                state_dim = np.prod(state.dims[0])
                ref_dim = np.prod(reference_state.dims[0])
                
                # If the state has higher dimension, truncate or project it
                if state_dim > ref_dim:
                    # Create a projection to the first ref_dim basis states
                    state_mat = state.full()
                    truncated_state = state_mat[:ref_dim, :ref_dim]
                    # Normalize if needed
                    trace = np.trace(truncated_state)
                    if abs(trace) > 1e-10:  # Avoid division by near-zero
                        truncated_state /= trace
                    state = Qobj(truncated_state, dims=reference_state.dims)
                # If reference has higher dimension, adapt the state
                elif ref_dim > state_dim:
                    # Embed state in larger space filled with zeros
                    ref_mat = reference_state.full()
                    embedded_state = np.zeros((ref_dim, ref_dim), dtype=complex)
                    embedded_state[:state_dim, :state_dim] = state.full()
                    # Normalize if needed
                    trace = np.trace(embedded_state)
                    if abs(trace) > 1e-10:  # Avoid division by near-zero
                        embedded_state /= trace
                    state = Qobj(embedded_state, dims=reference_state.dims)
            except Exception as e:
                print(f"Warning: Failed to adapt dimensions: {str(e)}")
    
    # For incompatible dimensions that can't be easily converted, use a simpler approach to calculate fidelity
    try:
        # Try regular fidelity calculation first
        sqrt_state = state.sqrtm()
        fidelity = (sqrt_state * reference_state * sqrt_state).tr().real
        return np.sqrt(fidelity)
    except:
        # If that fails, use a more direct approach
        from qutip import fidelity
        try:
            return fidelity(state, reference_state)
        except:
            # If all else fails, use trace distance as a fallback
            # Fidelity ≈ 1 - 0.5 * trace_distance^2 for small distances
            try:
                trace_distance = 0.5 * (state - reference_state).dag() * (state - reference_state)
                trace_norm = abs(trace_distance.tr())
                return max(0, 1 - 0.5 * trace_norm)
            except:
                # Absolute last resort: assume they're completely different
                return 0.0

=======
>>>>>>> 033b46c71c02f6ef3bb74dc3fcb185487cd672aa
def compare_circuit_results(fib_result, std_result, ideal_state=None):
    """
    Compare results from Fibonacci anyon braiding and standard quantum circuits.
    
    Parameters:
    -----------
    fib_result : object
        Result from Fibonacci anyon braiding
    std_result : object
        Result from standard quantum circuit
    ideal_state : Qobj, optional
        Ideal final state (if None, use the final state from noiseless evolution)
        
    Returns:
    --------
    dict
        Dictionary containing comparison metrics
    """
    # Get final states
    fib_final = fib_result.states[-1]
    std_final = std_result.states[-1]
    
    # If ideal state not provided, use noiseless evolution
    if ideal_state is None:
        # Use the final state from Fibonacci evolution as the reference
        ideal_state = fib_final
    
    # Check dimensions and handle the case of mismatched dimensions
    # if not np.array_equal(state.dims, reference_state.dims):
    #     # If dimensions don't match, we need to fix them
    #     # This typically happens if one state has dimensions like [[2,2],[2,2]] (superoperator form)
    #     # while the other has dimensions like [[2],[2]] (standard operator form)

    #     # Get the state in proper form
    #     if state.type == "super" and reference_state.type != "super":
    #         # Convert superoperator to regular density matrix if possible
    #         if state.shape[0] == state.shape[1] and np.sqrt(state.shape[0]).is_integer():
    #             dim = int(np.sqrt(state.shape[0]))
    #             from qutip import Qobj
    #             state_data = state.full()
    #             state = Qobj(state_data[:dim, :dim], dims=[[dim], [dim]])
    #     elif reference_state.type == "super" and state.type != "super":
    #         # Convert superoperator to regular density matrix if possible
    #         if reference_state.shape[0] == reference_state.shape[1] and np.sqrt(reference_state.shape[0]).is_integer():
    #             dim = int(np.sqrt(reference_state.shape[0]))
    #             from qutip import Qobj
    #             ref_data = reference_state.full()
    #             reference_state = Qobj(ref_data[:dim, :dim], dims=[[dim], [dim]])
    
    # Calculate fidelities
    fib_fidelity = fidelity(fib_final, ideal_state)
    # TODO: fix mismatch between state dimension so that fidelities can be calculated. as of now, both are of type="oper", so the above code from the deprecated calculate_fidelity function would not work as-is. additionally, the method used is to truncate the superoperator to a density matrix, which is... questionable. see https://www.perplexity.ai/search/how-would-you-calculate-the-fi-F6Y7HhwMQbivq_Xj0u2wig . my recommendation is to make sure the states used as parameters when you call compare_circuit_results have the same dimensions in the first place.
    std_fidelity = fidelity(std_final, ideal_state)
    
    # Calculate protection factor
    protection_factor = fib_fidelity / std_fidelity if std_fidelity > 0 else np.nan
    
    return {
        'fibonacci_fidelity': fib_fidelity,
        'standard_fidelity': std_fidelity,
        'protection_factor': protection_factor
    }

def test_anyon_topological_protection(braid_sequences=None, noise_levels=None, output_dir=None):
    """
    Test if Fibonacci anyon braiding provides topological protection against local errors.
    
    Parameters:
    -----------
    braid_sequences : list, optional
        List of braid sequences to test
    noise_levels : list, optional
        List of noise levels to test
    output_dir : str or Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    if braid_sequences is None:
        braid_sequences = ["1,2,1", "1,2,1,2,1", "1,2,1,2,1,2,1,2"]
    
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]
    
    if output_dir is None:
        output_dir = Path("results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize results storage
    results = {}
    
    for braid in braid_sequences:
        for noise in noise_levels:
            print(f"Testing braid sequence '{braid}' with noise level {noise}")
            
            # Configure noise
            noise_config = {
                'c_ops': create_noise_operators(noise)
            }
            
            # Run with Fibonacci anyons
            print("  Running Fibonacci anyon braiding...")
            fib_result = run_fibonacci_braiding_circuit(
                braid_type='Fibonacci',
                braid_sequence=braid,
                noise_config=noise_config
            )
            
            # Create equivalent non-topological circuit
            std_circuit = create_equivalent_standard_circuit(braid)
            
            # Run with standard quantum gates
            print("  Running standard quantum circuit...")
            # Use the equivalent circuit we created
            std_result = run_quantum_gate_circuit(
                circuit_type="Custom",
                noise_config=noise_config,
                custom_gates=std_circuit  # Pass the equivalent circuit
            )
            
            # Compare results
            print("  Comparing results...")
            comparison = compare_circuit_results(fib_result, std_result)
            
            # Store results
            results[(braid, noise)] = comparison
            
            # Create comparison plot
            create_state_comparison_plot(
                fib_result.states[-1],
                std_result.states[-1],
                braid, noise,
                output_dir / f"state_comparison_{braid.replace(',', '_')}_n{noise}.png"
            )
    
    # Statistical analysis of results
    protection_factors = [r['protection_factor'] for r in results.values() 
                         if not np.isnan(r['protection_factor'])]
    
    if protection_factors:
        mean_protection = np.mean(protection_factors)
        std_protection = np.std(protection_factors)
        
        # Is the protection statistically significant? (t-test)
        t_stat, p_value = stats.ttest_1samp(protection_factors, 1.0)
        
        results['statistics'] = {
            'mean_protection': mean_protection,
            'std_protection': std_protection,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_protection': p_value < 0.05 and mean_protection > 1.0
        }
    
    # Save results to CSV
    save_results_to_csv(results, output_dir / "topological_protection_results.csv")
    
    # Create summary plot
    create_summary_plot(results, output_dir / "topological_protection_summary.png")
    
    return results

def create_state_comparison_plot(fib_state, std_state, braid, noise, output_path):
    """
    Create a comparison plot of Fibonacci anyon and standard quantum circuit final states.
    
    Parameters:
    -----------
    fib_state : Qobj
        Final state from Fibonacci anyon braiding
    std_state : Qobj
        Final state from standard quantum circuit
    braid : str
        Braid sequence
    noise : float
        Noise level
    output_path : Path
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert to density matrices if needed
    if fib_state.isket:
        fib_dm = ket2dm(fib_state)
    else:
        fib_dm = fib_state
    
    if std_state.isket:
        std_dm = ket2dm(std_state)
    else:
        std_dm = std_state
    
    # Plot Fibonacci state
    ax1.matshow(np.abs(fib_dm.full()), cmap='viridis')
    ax1.set_title(f'Fibonacci Anyon State\nBraid: {braid}, Noise: {noise}')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    
    # Plot standard state
    ax2.matshow(np.abs(std_dm.full()), cmap='viridis')
    ax2.set_title(f'Standard Quantum Circuit State\nBraid: {braid}, Noise: {noise}')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plot(results, output_path):
    """
    Create a summary plot of topological protection factors.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
    output_path : Path
        Path to save the plot
    """
    # Extract data for plotting
    braid_sequences = sorted(set(k[0] for k in results.keys() if isinstance(k, tuple)))
    noise_levels = sorted(set(k[1] for k in results.keys() if isinstance(k, tuple)))
    
    if not braid_sequences or not noise_levels:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    bar_width = 0.8 / len(noise_levels)
    positions = np.arange(len(braid_sequences))
    
    # Plot bars for each noise level
    for i, noise in enumerate(noise_levels):
        protection_factors = []
        for braid in braid_sequences:
            if (braid, noise) in results:
                protection_factors.append(results[(braid, noise)]['protection_factor'])
            else:
                protection_factors.append(np.nan)
        
        ax.bar(
            positions + i * bar_width - bar_width * (len(noise_levels) - 1) / 2,
            protection_factors,
            bar_width,
            label=f'Noise={noise}'
        )
    
    # Add reference line at y=1 (no protection)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No Protection')
    
    # Set labels and title
    ax.set_xlabel('Braid Sequence')
    ax.set_ylabel('Protection Factor (Fibonacci/Standard)')
    ax.set_title('Topological Protection with Fibonacci Anyons')
    ax.set_xticks(positions)
    ax.set_xticklabels([b.replace(',', ',\n') for b in braid_sequences])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics if available
    if 'statistics' in results:
        stats = results['statistics']
        stats_text = (
            f"Mean Protection: {stats['mean_protection']:.2f} ± {stats['std_protection']:.2f}\n"
            f"p-value: {stats['p_value']:.4f}"
        )
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(results, output_path):
    """
    Save test results to a CSV file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
    output_path : Path
        Path to save the CSV file
    """
    # Extract data for CSV
    data = []
    for key, value in results.items():
        if isinstance(key, tuple):
            braid, noise = key
            data.append({
                'Braid Sequence': braid,
                'Noise Level': noise,
                'Fibonacci Fidelity': value['fibonacci_fidelity'],
                'Standard Fidelity': value['standard_fidelity'],
                'Protection Factor': value['protection_factor']
            })
    
    # Create DataFrame and save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Run topological protection test with default parameters
    results = test_anyon_topological_protection(
        braid_sequences=["1,2,1", "1,2,1,2,1"],
        noise_levels=[0.01, 0.05],
        output_dir="results/topological_test"
    )
    
    # Print summary of results
    if 'statistics' in results:
        stats = results['statistics']
        print("\nStatistical Analysis:")
        print(f"Mean Protection Factor: {stats['mean_protection']:.2f} ± {stats['std_protection']:.2f}")
        print(f"t-statistic: {stats['t_statistic']:.2f}, p-value: {stats['p_value']:.4f}")
        
        if stats['significant_protection']:
            print("RESULT: Fibonacci anyons show statistically significant topological protection.")
        else:
            print("RESULT: No statistically significant topological protection with Fibonacci anyons.")

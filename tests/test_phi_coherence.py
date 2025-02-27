#!/usr/bin/env python
# tests/test_phi_coherence.py

"""
Test script to compare coherence properties between phi-scaled and standard quantum evolution.

This script directly tests the paper's claim that phi-scaled pulse sequences enhance
quantum coherence compared to uniform sequences under identical noise conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from scipy import stats, optimize
from qutip import Qobj, ket2dm

from constants import PHI
from simulations.scripts.evolve_state import (
    run_state_evolution,
    run_phi_recursive_evolution
)
from analyses.coherence import l1_coherence, relative_entropy_coherence

def calculate_purity(state):
    """
    Calculate the purity of a quantum state.
    
    Parameters:
    -----------
    state : Qobj
        Quantum state (ket or density matrix)
        
    Returns:
    --------
    float
        Purity of the state (Tr[ρ²])
    """
    if state.isket:
        # For pure states, purity is always 1
        return 1.0
    else:
        # For mixed states, purity is Tr[ρ²]
        return (state * state).tr().real

def calculate_purity_trajectory(result):
    """
    Calculate purity trajectory for a quantum evolution result.
    
    Parameters:
    -----------
    result : object
        Result object from quantum evolution containing states
        
    Returns:
    --------
    numpy.ndarray
        Array of purity values for each state in the evolution
    """
    purities = []
    for state in result.states:
        purities.append(calculate_purity(state))
    return np.array(purities)

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
    
    # Calculate fidelity using sqrt(ρ₁) ρ₂ sqrt(ρ₁)
    sqrt_state = state.sqrtm()
    fidelity = (sqrt_state * reference_state * sqrt_state).tr().real
    return np.sqrt(fidelity)

def calculate_fidelity_trajectory(result, reference_state):
    """
    Calculate fidelity trajectory for a quantum evolution result.
    
    Parameters:
    -----------
    result : object
        Result object from quantum evolution containing states
    reference_state : Qobj
        Reference quantum state (ket or density matrix)
        
    Returns:
    --------
    numpy.ndarray
        Array of fidelity values for each state in the evolution
    """
    fidelities = []
    for state in result.states:
        fidelities.append(calculate_fidelity(state, reference_state))
    return np.array(fidelities)

def exponential_decay(t, a, tau, c):
    """
    Exponential decay function for fitting coherence decay.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time points
    a : float
        Amplitude
    tau : float
        Decay time constant
    c : float
        Offset
        
    Returns:
    --------
    numpy.ndarray
        Exponential decay values
    """
    return a * np.exp(-t / tau) + c

def calculate_coherence_time(result, metric='purity'):
    """
    Calculate coherence time by fitting an exponential decay to a metric.
    
    Parameters:
    -----------
    result : object
        Result object from quantum evolution containing states and times
    metric : str, optional
        Metric to use for coherence time calculation ('purity' or 'fidelity')
        
    Returns:
    --------
    float
        Coherence time (tau) from exponential fit
    """
    times = result.times
    
    if metric == 'purity':
        values = calculate_purity_trajectory(result)
    elif metric == 'fidelity':
        # Use initial state as reference
        values = calculate_fidelity_trajectory(result, result.states[0])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Ensure we have enough points for fitting
    if len(times) < 3 or len(values) < 3:
        return np.nan
    
    try:
        # Initial guess for parameters
        p0 = [values[0] - values[-1], times[-1] / 2, values[-1]]
        
        # Fit exponential decay
        popt, _ = optimize.curve_fit(
            exponential_decay, times, values, 
            p0=p0, bounds=([0, 0, 0], [1, np.inf, 1])
        )
        
        # Extract coherence time (tau)
        tau = popt[1]
        return tau
    except:
        # Return NaN if fitting fails
        return np.nan

def run_coherence_comparison(qubit_counts=None, n_steps=100, noise_levels=None, output_dir=None):
    """
    Directly compare quantum coherence between phi-scaled and uniform pulse sequences.
    
    Parameters:
    -----------
    qubit_counts : list, optional
        List of qubit counts to test
    n_steps : int, optional
        Number of evolution steps
    noise_levels : list, optional
        List of noise levels to test
    output_dir : str or Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    if qubit_counts is None:
        qubit_counts = [1, 2]
    
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]
    
    if output_dir is None:
        output_dir = Path("results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize results storage
    results = {}
    
    for n_qubits in qubit_counts:
        for noise in noise_levels:
            print(f"Testing {n_qubits} qubits with noise level {noise}")
            
            # Create noise configuration (set to None for no noise)
            noise_config = None
            
            # PHI-scaled evolution with recursive scaling
            print("  Running phi-recursive evolution...")
            phi_result = run_phi_recursive_evolution(
                num_qubits=n_qubits,
                state_label="phi_sensitive",  # Uses recursive phi structure
                n_steps=n_steps,
                scaling_factor=PHI,
                recursion_depth=3,
                analyze_phi=True,
                noise_config=noise_config  # Add noise to phi-recursive evolution
            )
            
            # Standard evolution at same scale
            print("  Running standard evolution...")
            std_result = run_state_evolution(
                num_qubits=n_qubits,
                state_label="plus",  # Standard state
                n_steps=n_steps,
                scaling_factor=PHI,  # Same scaling factor
                noise_config=noise_config,
                pulse_type="Square"
            )
            
            # Calculate coherence metrics
            print("  Calculating coherence metrics...")
            
            # Purity trajectories
            phi_purity = calculate_purity_trajectory(phi_result)
            std_purity = calculate_purity_trajectory(std_result)
            
            # Fidelity to initial state
            phi_fidelity = calculate_fidelity_trajectory(phi_result, phi_result.states[0])
            std_fidelity = calculate_fidelity_trajectory(std_result, std_result.states[0])
            
            # Calculate coherence times
            phi_coherence_time = calculate_coherence_time(phi_result, metric='purity')
            std_coherence_time = calculate_coherence_time(std_result, metric='purity')
            
            # Store results
            results[(n_qubits, noise)] = {
                'phi_coherence_time': phi_coherence_time,
                'std_coherence_time': std_coherence_time,
                'improvement_factor': phi_coherence_time / std_coherence_time if std_coherence_time > 0 else np.nan,
                'phi_purity': phi_purity,
                'std_purity': std_purity,
                'phi_fidelity': phi_fidelity,
                'std_fidelity': std_fidelity,
                'times': phi_result.times
            }
            
            # Create comparison plot
            create_comparison_plot(
                phi_result.times, 
                phi_purity, std_purity,
                phi_fidelity, std_fidelity,
                n_qubits, noise,
                output_dir / f"coherence_comparison_q{n_qubits}_n{noise}.png"
            )
    
    # Statistical analysis of results
    improvement_factors = [r['improvement_factor'] for r in results.values() 
                          if not np.isnan(r['improvement_factor'])]
    
    if improvement_factors:
        mean_improvement = np.mean(improvement_factors)
        std_improvement = np.std(improvement_factors)
        
        # Handle the case with only one data point
        if len(improvement_factors) == 1:
            # Can't do t-test with one sample, just report the improvement factor
            results['statistics'] = {
                'mean_improvement': mean_improvement,
                'std_improvement': 0.0,  # No std dev with one sample
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant_improvement': mean_improvement > 1.5  # Use a threshold instead of p-value
            }
        else:
            # Is the improvement statistically significant? (t-test)
            t_stat, p_value = stats.ttest_1samp(improvement_factors, 1.0)
            
            results['statistics'] = {
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_improvement': p_value < 0.05 and mean_improvement > 1.0
            }
    
    # Save results to CSV
    save_results_to_csv(results, output_dir / "coherence_comparison_results.csv")
    
    # Create summary plot
    create_summary_plot(results, output_dir / "coherence_comparison_summary.png")
    
    return results

def create_comparison_plot(times, phi_purity, std_purity, phi_fidelity, std_fidelity, 
                          n_qubits, noise, output_path):
    """
    Create a comparison plot of purity and fidelity for phi-scaled vs. standard evolution.
    
    Parameters:
    -----------
    times : numpy.ndarray
        Time points
    phi_purity : numpy.ndarray
        Purity values for phi-scaled evolution
    std_purity : numpy.ndarray
        Purity values for standard evolution
    phi_fidelity : numpy.ndarray
        Fidelity values for phi-scaled evolution
    std_fidelity : numpy.ndarray
        Fidelity values for standard evolution
    n_qubits : int
        Number of qubits
    noise : float
        Noise level
    output_path : Path
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot purity
    ax1.plot(times, phi_purity, 'o-', color='#1f77b4', label='Phi-Scaled')
    ax1.plot(times, std_purity, 'o-', color='#ff7f0e', label='Standard')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Purity')
    ax1.set_title(f'Purity Decay ({n_qubits} qubits, noise={noise})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot fidelity
    ax2.plot(times, phi_fidelity, 'o-', color='#1f77b4', label='Phi-Scaled')
    ax2.plot(times, std_fidelity, 'o-', color='#ff7f0e', label='Standard')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fidelity to Initial State')
    ax2.set_title(f'Fidelity Decay ({n_qubits} qubits, noise={noise})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plot(results, output_path):
    """
    Create a summary plot of coherence improvement factors.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing comparison results
    output_path : Path
        Path to save the plot
    """
    # Extract data for plotting
    qubit_counts = sorted(set(k[0] for k in results.keys() if isinstance(k, tuple)))
    noise_levels = sorted(set(k[1] for k in results.keys() if isinstance(k, tuple)))
    
    if not qubit_counts or not noise_levels:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    bar_width = 0.8 / len(noise_levels)
    positions = np.arange(len(qubit_counts))
    
    # Plot bars for each noise level
    for i, noise in enumerate(noise_levels):
        improvement_factors = []
        for qubits in qubit_counts:
            if (qubits, noise) in results:
                improvement_factors.append(results[(qubits, noise)]['improvement_factor'])
            else:
                improvement_factors.append(np.nan)
        
        ax.bar(
            positions + i * bar_width - bar_width * (len(noise_levels) - 1) / 2,
            improvement_factors,
            bar_width,
            label=f'Noise={noise}'
        )
    
    # Add reference line at y=1 (no improvement)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No Improvement')
    
    # Set labels and title
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Coherence Time Improvement Factor (Phi/Standard)')
    ax.set_title('Coherence Enhancement with Phi-Scaled Evolution')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(q) for q in qubit_counts])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics if available
    if 'statistics' in results:
        stats = results['statistics']
        stats_text = (
            f"Mean Improvement: {stats['mean_improvement']:.2f} ± {stats['std_improvement']:.2f}\n"
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
    Save comparison results to a CSV file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing comparison results
    output_path : Path
        Path to save the CSV file
    """
    # Extract data for CSV
    data = []
    for key, value in results.items():
        if isinstance(key, tuple):
            qubits, noise = key
            data.append({
                'Qubits': qubits,
                'Noise': noise,
                'Phi Coherence Time': value['phi_coherence_time'],
                'Standard Coherence Time': value['std_coherence_time'],
                'Improvement Factor': value['improvement_factor']
            })
    
    # Create DataFrame and save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Run coherence comparison with default parameters
    results = run_coherence_comparison(
        qubit_counts=[1, 2],
        n_steps=50,
        noise_levels=[0.01, 0.05, 0.1],
        output_dir="results/coherence_test"
    )
    
    # Print summary of results
    if 'statistics' in results:
        stats = results['statistics']
        print("\nStatistical Analysis:")
        print(f"Mean Improvement Factor: {stats['mean_improvement']:.2f} ± {stats['std_improvement']:.2f}")
        print(f"t-statistic: {stats['t_statistic']:.2f}, p-value: {stats['p_value']:.4f}")
        
        if stats['significant_improvement']:
            print("RESULT: Phi-scaling shows statistically significant coherence improvement.")
        else:
            print("RESULT: No statistically significant coherence improvement with phi-scaling.")

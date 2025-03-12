#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze how key quantum properties scale with the f_s parameter.

This script runs simulations with different f_s values and analyzes how properties
such as band gap size, fractal dimension, and topological invariants change.
Results are presented in both tabular and graphical formats.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from qutip import Qobj, tensor, sigmaz, sigmax, qeye, basis
from pathlib import Path

# Import simulation components
from simulations.scripts.evolve_circuit import run_phi_scaled_twoqubit_circuit
from analyses.fractal_analysis import compute_energy_spectrum, estimate_fractal_dimension
from analyses.topological_invariants import compute_winding_number
from constants import PHI

def analyze_fs_scaling(fs_values=None, save_results=True):
    """
    Analyze how quantum properties scale with different f_s values.
    
    Parameters:
    -----------
    fs_values : list, optional
        List of f_s values to analyze. If None, uses default values.
    save_results : bool, optional
        Whether to save results to files.
        
    Returns:
    --------
    dict
        Dictionary containing analysis results.
    """
    if fs_values is None:
        # Use a range of values including PHI and rational approximations
        fs_values = [0.5, 1.0, PHI, 2.0, 2.5, 3.0]
    
    # Initialize results storage
    results = {
        'fs_values': fs_values,
        'band_gaps': [],
        'fractal_dimensions': [],
        'topological_invariants': [],
        'correlation_lengths': []
    }
    
    # Create k-points for topological analysis
    k_points = np.linspace(0, 2*np.pi, 100)
    
    print(f"Analyzing {len(fs_values)} f_s values: {fs_values}")
    
    # Run simulations for each f_s value
    for i, fs in enumerate(fs_values):
        print(f"Processing f_s = {fs:.4f} ({i+1}/{len(fs_values)})")
        
        # Run simulation with current f_s
        result = run_phi_scaled_twoqubit_circuit(scaling_factor=fs)
        
        # Extract Hamiltonian function
        H_func = result.hamiltonian
        
        # Compute energy spectrum
        f_s_sweep, energies, spectrum_analysis = compute_energy_spectrum(
            H_func, 
            config={'energy_spectrum': {'f_s_range': [0.0, 5.0], 'resolution': 200}}
        )
        
        # Calculate band gap (difference between lowest eigenvalues)
        # Take the minimum non-zero gap as the characteristic gap size
        gaps = []
        for e in energies:
            if len(e) > 1:  # Ensure we have at least 2 eigenvalues
                sorted_e = np.sort(e)
                # Find gaps between adjacent eigenvalues
                level_gaps = np.diff(sorted_e)
                # Filter out very small gaps (numerical artifacts)
                significant_gaps = level_gaps[level_gaps > 1e-10]
                if len(significant_gaps) > 0:
                    gaps.append(np.min(significant_gaps))
        
        # Use median gap size for robustness
        if gaps:
            band_gap = np.median(gaps)
        else:
            band_gap = np.nan
        
        # Compute fractal dimension from energy spectrum
        # Reshape energies to 1D array for fractal analysis
        flat_energies = np.array(energies).flatten()
        fractal_dim, dim_info = estimate_fractal_dimension(flat_energies)
        
        # Create eigenstates for topological analysis
        # We'll use a simplified approach to generate eigenstates from the Hamiltonian
        eigenstates = []
        for k in k_points:
            # Scale Hamiltonian by k
            H_k = fs * (tensor(sigmaz(), qeye(2)) + k * tensor(qeye(2), sigmax()))
            # Get eigenstates
            _, states = H_k.eigenstates()
            eigenstates.append(states[0])  # Use ground state
        
        # Compute winding number as topological invariant
        winding = compute_winding_number(eigenstates, k_points)
        
        # Estimate correlation length from energy spectrum
        # Use inverse of minimum gap as proxy for correlation length
        if band_gap > 1e-10:
            correlation_length = 1.0 / band_gap
        else:
            correlation_length = np.nan
        
        # Store results
        results['band_gaps'].append(band_gap)
        results['fractal_dimensions'].append(fractal_dim)
        results['topological_invariants'].append(winding)
        results['correlation_lengths'].append(correlation_length)
    
    # Convert results to DataFrame for easier handling
    df = pd.DataFrame({
        'f_s': results['fs_values'],
        'Band Gap': results['band_gaps'],
        'Fractal Dimension': results['fractal_dimensions'],
        'Topological Invariant': results['topological_invariants'],
        'Correlation Length': results['correlation_lengths']
    })
    
    # Print results table
    print("\nResults Table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    if save_results:
        # Save results to CSV
        output_dir = Path(".")
        df.to_csv(output_dir / "fs_scaling_results.csv", index=False)
        print(f"Results saved to fs_scaling_results.csv")
        
        # Create visualization
        create_fs_scaling_plots(results, output_dir)
    
    return results

def create_fs_scaling_plots(results, output_dir=None):
    """
    Create plots showing how properties scale with f_s.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results.
    output_dir : Path, optional
        Directory to save plots. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = Path(".")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot band gap vs f_s
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['fs_values'], results['band_gaps'], 'o-', color='#1f77b4', linewidth=2)
    ax1.set_xlabel('Scale Factor (f_s)')
    ax1.set_ylabel('Band Gap Size')
    ax1.set_title('Band Gap vs. Scale Factor')
    ax1.grid(True, alpha=0.3)
    
    # Plot fractal dimension vs f_s
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2)
    ax2.set_xlabel('Scale Factor (f_s)')
    ax2.set_ylabel('Fractal Dimension')
    ax2.set_title('Fractal Dimension vs. Scale Factor')
    ax2.grid(True, alpha=0.3)
    
    # Plot topological invariant vs f_s
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['fs_values'], results['topological_invariants'], 'o-', color='#2ca02c', linewidth=2)
    ax3.set_xlabel('Scale Factor (f_s)')
    ax3.set_ylabel('Topological Invariant')
    ax3.set_title('Topological Invariant vs. Scale Factor')
    ax3.grid(True, alpha=0.3)
    
    # Plot correlation length vs f_s
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['fs_values'], results['correlation_lengths'], 'o-', color='#d62728', linewidth=2)
    ax4.set_xlabel('Scale Factor (f_s)')
    ax4.set_ylabel('Correlation Length')
    ax4.set_title('Correlation Length vs. Scale Factor')
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('f_s-Driven Quantum Properties', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / "fs_scaling_plots.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to fs_scaling_plots.png")
    
    # Create combined plot with all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize each metric to [0,1] for comparison
    def normalize(data):
        data = np.array(data)
        if np.all(np.isnan(data)) or np.max(data) == np.min(data):
            return np.zeros_like(data)
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    
    norm_gaps = normalize(results['band_gaps'])
    norm_dims = normalize(results['fractal_dimensions'])
    norm_topos = normalize(results['topological_invariants'])
    norm_corrs = normalize(results['correlation_lengths'])
    
    # Plot normalized metrics
    ax.plot(results['fs_values'], norm_gaps, 'o-', color='#1f77b4', linewidth=2, label='Band Gap')
    ax.plot(results['fs_values'], norm_dims, 's-', color='#ff7f0e', linewidth=2, label='Fractal Dimension')
    ax.plot(results['fs_values'], norm_topos, '^-', color='#2ca02c', linewidth=2, label='Topological Invariant')
    ax.plot(results['fs_values'], norm_corrs, 'D-', color='#d62728', linewidth=2, label='Correlation Length')
    
    # Add vertical line at PHI
    ax.axvline(x=PHI, color='k', linestyle='--', alpha=0.5, label=f'φ ≈ {PHI:.4f}')
    
    ax.set_xlabel('Scale Factor (f_s)')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Quantum Properties vs. Scale Factor')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save combined figure
    plt.savefig(output_dir / "fs_scaling_combined.png", dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to fs_scaling_combined.png")
    
    plt.close('all')

if __name__ == "__main__":
    # Define f_s values to analyze
    # Include PHI and rational approximations of PHI
    phi_approx = [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]  # Fibonacci sequence ratios
    fs_values = [ratio[0]/ratio[1] for ratio in phi_approx]
    fs_values.append(PHI)  # Add exact PHI
    fs_values.extend([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # Add other values
    fs_values = sorted(list(set(fs_values)))  # Remove duplicates and sort
    
    # Run analysis
    analyze_fs_scaling(fs_values)
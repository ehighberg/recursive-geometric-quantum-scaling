#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the significance of the golden ratio (phi) in quantum simulations.

This script compares quantum properties at phi with those at other scaling factors,
focusing on identifying special behavior or phase transitions near phi.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from qutip import Qobj, tensor, sigmaz, sigmax, qeye, basis
from pathlib import Path

# Import simulation components
from simulations.scripts.evolve_circuit import run_phi_scaled_twoqubit_circuit
from simulations.scripts.evolve_state import run_state_evolution
from analyses.fractal_analysis import compute_energy_spectrum, estimate_fractal_dimension
from analyses.topological_invariants import compute_phi_sensitive_winding, compute_phi_sensitive_z2, compute_phi_resonant_berry_phase
from constants import PHI

def analyze_phi_significance(fine_resolution=True, save_results=True):
    """
    Analyze the significance of phi in quantum simulations by comparing
    properties at phi with those at nearby values.
    
    Parameters:
    -----------
    fine_resolution : bool, optional
        Whether to use fine resolution around phi.
    save_results : bool, optional
        Whether to save results to files.
        
    Returns:
    --------
    dict
        Dictionary containing analysis results.
    """
    # Define scaling factors to analyze
    if fine_resolution:
        # Create a fine grid around phi
        phi_neighborhood = np.linspace(PHI - 0.2, PHI + 0.2, 21)
        # Add broader range for context
        fs_values = np.concatenate([
            np.linspace(0.5, PHI - 0.2, 10),
            phi_neighborhood,
            np.linspace(PHI + 0.2, 3.0, 10)
        ])
    else:
        # Use coarser grid
        fs_values = np.linspace(0.5, 3.0, 26)
    
    # Add exact phi value
    fs_values = np.sort(np.append(fs_values, PHI))
    
    # Initialize results storage
    results = {
        'fs_values': fs_values,
        'band_gaps': np.zeros_like(fs_values),
        'fractal_dimensions': np.zeros_like(fs_values),
  # Standard fractal dimensions
        'topological_invariants': np.zeros_like(fs_values),
  # Using proper topological definition
        'correlation_lengths': np.zeros_like(fs_values),
  # Correlation length without phi-bias
        'berry_phases': np.zeros_like(fs_values),  # Added: Track Berry phases
        'energy_spectra': [],
        'wavefunction_profiles': []
    }
    
    print(f"Analyzing {len(fs_values)} f_s values with phi = {PHI:.6f}")
    
    # Run simulations for each f_s value
    for i, fs in enumerate(fs_values):
        print(f"Processing f_s = {fs:.6f} ({i+1}/{len(fs_values)})")
        
        # Run circuit simulation with current f_s
        circuit_result = run_phi_scaled_twoqubit_circuit(scaling_factor=fs)
        
        # Also run state evolution for comparison
        state_result = run_state_evolution(
            num_qubits=1,
            state_label="plus",
            n_steps=50,
            scaling_factor=fs
        )
        
        # Extract Hamiltonian function
        H_func = circuit_result.hamiltonian
        
        # Compute energy spectrum
        f_s_sweep, energies, spectrum_analysis = compute_energy_spectrum(
            H_func, 
            config={'energy_spectrum': {'f_s_range': [0.0, 5.0], 'resolution': 100}}
        )
        
        # Store energy spectrum for this f_s value
        results['energy_spectra'].append(energies)
        
        # Calculate band gap
        gaps = []
        for e in energies:
            if len(e) > 1:
                sorted_e = np.sort(e)
                level_gaps = np.diff(sorted_e)
                significant_gaps = level_gaps[level_gaps > 1e-10]
                if len(significant_gaps) > 0:
                    gaps.append(np.min(significant_gaps))
        
        if gaps:
            band_gap = np.median(gaps)
        else:
            band_gap = np.nan
        
        # Compute fractal dimension from energy spectrum
        # Calculate fractal dimension for each energy level separately
        fractal_dims = []
        for level_energies in energies:
            if len(level_energies) > 10:  # Ensure enough points for meaningful calculation
                dim, info = estimate_fractal_dimension(level_energies)
                if not np.isnan(dim):
                    fractal_dims.append(dim)

        
# Use standard fractal dimension calculation with proper mathematical definition
        # Use the mean of the individual dimensions if available
        if fractal_dims:
            fractal_dim = np.nanmean(fractal_dims)
        else:
            # Fallback to flattened approach if individual calculations fail
            flat_energies = np.array(energies).flatten()
            fractal_dim, dim_info = estimate_fractal_dimension(flat_energies)
        
    
        # Create k-points for topological analysis
        k_points = np.linspace(0, 2*np.pi, 100)
        
        # Create eigenstates for topological analysis
        eigenstates = []
        for k in k_points:
            H_k = fs * (tensor(sigmaz(), qeye(2)) + k * tensor(qeye(2), sigmax()))
            _, states = H_k.eigenstates()
            eigenstates.append(states[0])
        
        # Use phi-sensitive versions of topological invariants that maintain mathematical rigor
        # These calculate proper invariants without artificial phi-based modifications
        winding = compute_phi_sensitive_winding(eigenstates, k_points, fs)
        
        # Compute Z2 index as alternative topological invariant
        z2_index = compute_phi_sensitive_z2(eigenstates, k_points, fs)
        
        # Calculate proper Berry phase for the path through k-space
        berry_phase = compute_phi_resonant_berry_phase(eigenstates, fs)
        
        # Estimate correlation length
        if band_gap > 1e-10:
            correlation_length = 1.0 / band_gap
        else:
            correlation_length = np.nan
        
        # Store wavefunction profile from state evolution
        if hasattr(state_result, 'wavefunction'):
            results['wavefunction_profiles'].append(state_result.wavefunction)
        else:
            results['wavefunction_profiles'].append(None)
        
        # Store results
        results['band_gaps'][i] = band_gap
        results['fractal_dimensions'][i] = fractal_dim
        results['topological_invariants'][i] = winding
        results['correlation_lengths'][i] = correlation_length
        results['berry_phases'][i] = berry_phase
    
    # Convert results to DataFrame for easier handling
    df = pd.DataFrame({
        'f_s': results['fs_values'],
        'Band Gap': results['band_gaps'],
        'Fractal Dimension': results['fractal_dimensions'],
        'Topological Invariant': results['topological_invariants'],
        'Correlation Length': results['correlation_lengths']
,
        'Berry Phase': results['berry_phases']
    })
    
    # Add column indicating if value is phi
    df['Is Phi'] = np.isclose(df['f_s'], PHI, rtol=1e-10)
    
    # Print results table
    print("\nResults Table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    if save_results:
        # Save results to CSV
        output_dir = Path(".")
        df.to_csv(output_dir / "phi_significance_results.csv", index=False)
        print(f"Results saved to phi_significance_results.csv")
        
        # Create visualization
        create_phi_significance_plots(results, output_dir)
    
    return results

def create_phi_significance_plots(results, output_dir=None):
    """
    Create plots showing the significance of phi in quantum properties.
    
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
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot band gap vs f_s with phi highlighted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['fs_values'], results['band_gaps'], 'o-', color='#1f77b4', linewidth=2)
    ax1.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax1.set_xlabel('Scale Factor (f_s)')
    ax1.set_ylabel('Band Gap Size')
    ax1.set_title('Band Gap vs. Scale Factor')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot fractal dimension vs f_s with phi highlighted
    # Using mathematically sound fractal dimensions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2, label='Fractal Dimension')
    ax2.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax2.set_xlabel('Scale Factor (f_s)')
    ax2.set_ylabel('Fractal Dimension')
    ax2.set_title('Fractal Dimension vs. Scale Factor')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot topological invariant vs f_s with phi highlighted
 
    # Using mathematically sound topological invariants
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['fs_values'], results['topological_invariants'], 'o-', color='#2ca02c', linewidth=2, label='Winding Number')
    ax3.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax3.set_xlabel('Scale Factor (f_s)')
    ax3.set_ylabel('Topological Invariant')
    ax3.set_title('Topological Invariant vs. Scale Factor')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot correlation length vs f_s with phi highlighted
    # Using mathematically sound correlation length calculation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['fs_values'], results['correlation_lengths'], 'o-', color='#d62728', linewidth=2, label='Correlation Length')
    ax4.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax4.plot(results['fs_values'], results['berry_phases']/(2*np.pi), 'o--', color='#9467bd', linewidth=1, label='Berry Phase/(2π)')
    ax4.set_xlabel('Scale Factor (f_s)')
    ax4.set_ylabel('Correlation Length')
    ax4.set_title('Correlation Length vs. Scale Factor')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add overall title
    fig.suptitle('Analysis of Quantum Properties with Different Scaling Factors', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / "phi_significance_plots.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to phi_significance_plots.png")
    
    # Create derivative plots to identify phase transitions
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate numerical derivatives
    fs_values = results['fs_values']
    h = np.diff(fs_values)
    
    # Function to calculate smoothed numerical derivative
    def smooth_derivative(y, window_length=5, polyorder=2):
        """
        Calculate smoothed derivative using Savitzky-Golay filter.
        
        Parameters:
        -----------
        y : numpy.ndarray
            Input data
        window_length : int, optional
            Length of the filter window (must be odd)
        polyorder : int, optional
            Order of the polynomial used for filtering
            
        Returns:
        --------
        numpy.ndarray
            Smoothed derivative
        """
        from scipy.signal import savgol_filter
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window_length is not larger than data length
        if window_length > len(y):
            window_length = min(len(y) - (len(y) % 2 == 0), 5)
            polyorder = min(polyorder, window_length - 1)
        
        if len(y) < window_length:
            # Not enough points for smoothing, use simple differences
            if len(y) != len(fs_values):
                return np.zeros_like(fs_values)
            dy = np.diff(y)
            derivative = np.zeros_like(fs_values)
            derivative[:-1] = dy / h
            derivative[-1] = derivative[-2]  # Extend last value
            return derivative
        
        try:
            # First smooth the data
            y_smooth = savgol_filter(y, window_length, polyorder)
            
            # Then calculate derivative
            dy = np.diff(y_smooth)
            derivative = np.zeros_like(fs_values)
            derivative[:-1] = dy / h
            derivative[-1] = derivative[-2]  # Extend last value
            return derivative
        except Exception as e:
            print(f"Warning: Smoothing failed with error: {e}. Using simple differences.")
            # Fallback to simple differences
            if len(y) != len(fs_values):
                return np.zeros_like(fs_values)
            dy = np.diff(y)
            derivative = np.zeros_like(fs_values)
            derivative[:-1] = dy / h
            derivative[-1] = derivative[-2]  # Extend last value
            return derivative
    
    # Calculate derivatives using the smooth derivative function
    d_gap = smooth_derivative(results['band_gaps'])
    d_dim = smooth_derivative(results['fractal_dimensions'])
    d_topo = smooth_derivative(results['topological_invariants'])
    d_corr = smooth_derivative(results['correlation_lengths'])
    
    # Plot derivatives
    axs[0, 0].plot(fs_values, d_gap, 'o-', color='#1f77b4')
    axs[0, 0].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[0, 0].set_title('d(Band Gap)/d(f_s)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    axs[0, 1].plot(fs_values, d_dim, 'o-', color='#ff7f0e')
    axs[0, 1].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[0, 1].set_title('d(Fractal Dimension)/d(f_s)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    axs[1, 0].plot(fs_values, d_topo, 'o-', color='#2ca02c')
    axs[1, 0].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[1, 0].set_title('d(Topological Invariant)/d(f_s)')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    axs[1, 1].plot(fs_values, d_corr, 'o-', color='#d62728')
    axs[1, 1].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[1, 1].set_title('d(Correlation Length)/d(f_s)')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()
    
    fig.suptitle('Derivatives of Quantum Properties (Phase Transition Indicators)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
 
    
    # Save derivatives figure
    plt.savefig(output_dir / "phi_significance_derivatives.png", dpi=300, bbox_inches='tight')
    print(f"Derivative plots saved to phi_significance_derivatives.png")
    
    # Create zoom-in plot around phi
    phi_idx = np.argmin(np.abs(fs_values - PHI))
    window = 10  # Points on each side of phi
    
    # Ensure we have enough points
    left_idx = max(0, phi_idx - window)
    right_idx = min(len(fs_values) - 1, phi_idx + window)
    
    zoom_fs = fs_values[left_idx:right_idx+1]
    zoom_gaps = results['band_gaps'][left_idx:right_idx+1]
    zoom_dims = results['fractal_dimensions'][left_idx:right_idx+1]
    zoom_topos = results['topological_invariants'][left_idx:right_idx+1]
    zoom_corrs = results['correlation_lengths'][left_idx:right_idx+1]
    zoom_berry = results['berry_phases'][left_idx:right_idx+1]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].plot(zoom_fs, zoom_gaps, 'o-', color='#1f77b4', linewidth=2)
    axs[0, 0].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[0, 0].set_title('Band Gap (Zoom Around φ)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    axs[0, 1].plot(zoom_fs, zoom_dims, 'o-', color='#ff7f0e', linewidth=2)
    axs[0, 1].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[0, 1].set_title('Fractal Dimension (Zoom Around φ)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    axs[1, 0].plot(zoom_fs, zoom_topos, 'o-', color='#2ca02c', linewidth=2)
    axs[1, 0].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[1, 0].set_title('Topological Invariant (Zoom Around φ)')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()
    
    axs[1, 1].plot(zoom_fs, zoom_corrs, 'o-', color='#d62728', linewidth=2)
    axs[1, 1].axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    axs[1, 1].plot(zoom_fs, zoom_berry/(2*np.pi), 'o--', color='#9467bd', linewidth=1, label='Berry Phase/(2π)')
    axs[1, 1].set_title('Correlation Length and Berry Phase (Zoom Around φ)')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()
    
    fig.suptitle('Zoom View of Quantum Properties Around φ', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save zoom figure
    plt.savefig(output_dir / "phi_significance_zoom.png", dpi=300, bbox_inches='tight')
    print(f"Zoom plots saved to phi_significance_zoom.png")
    
    plt.close('all')

if __name__ == "__main__":
    # Run analysis with fine resolution around phi
    analyze_phi_significance(fine_resolution=True)

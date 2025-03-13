<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing the statistical significance of the golden ratio (phi) in quantum metrics.

This module provides tools for assessing whether observations at the golden ratio (phi)
are statistically different from other scaling factors, providing methods to test
the phi-resonance hypothesis in quantum systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# Define phi constant
PHI = (1 + np.sqrt(5)) / 2

def analyze_phi_significance(
    scaling_factors: List[float],
    metrics: Dict[str, Dict[float, np.ndarray]],
    phi_value: float = PHI,
    output_dir: Optional[str] = None,
    plot: bool = True,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze the statistical significance of phi in quantum metrics.
    
    Parameters:
        scaling_factors (List[float]): List of scaling factors used
        metrics (Dict[str, Dict[float, np.ndarray]]): Dictionary mapping metric names to
                                                     nested dictionaries that map scaling factors
                                                     to data arrays
        phi_value (float): Value of phi to use for comparison (default is the golden ratio)
        output_dir (Optional[str]): Directory to save results
        plot (bool): Whether to generate plots
        alpha (float): Significance level for statistical tests
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    # Ensure phi is in the scaling factors list
    if phi_value not in scaling_factors:
        raise ValueError(f"Phi value {phi_value} not found in scaling factors list")
    
    results = {}
    p_values = []
    
    # Compare each metric's values at phi with other scaling factors
    for metric_name, metric_data in metrics.items():
        if phi_value not in metric_data:
            continue
            
        phi_data = metric_data[phi_value]
        
        # Combine data from other scaling factors for comparison
        other_data = []
        for factor in scaling_factors:
            if factor != phi_value and factor in metric_data:
                other_data.append(metric_data[factor])
        
        if not other_data:
            continue
            
        combined_other = np.concatenate(other_data)
        
        # Calculate basic statistics
        phi_mean = np.mean(phi_data)
        phi_std = np.std(phi_data, ddof=1)
        other_mean = np.mean(combined_other)
        other_std = np.std(combined_other, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(phi_data, combined_other, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        effect_size = calculate_cohens_d(phi_data, combined_other)
        
        # Store p-value for multiple testing correction
        p_values.append((metric_name, p_value))
        
        # Store results
        metric_result = {
            'phi_mean': phi_mean,
            'phi_std': phi_std,
            'other_mean': other_mean,
            'other_std': other_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < alpha
        }
        
        # Calculate percent difference and z-score
        if other_mean != 0:
            metric_result['percent_difference'] = 100 * (phi_mean - other_mean) / other_mean
        else:
            metric_result['percent_difference'] = np.nan
            
        if other_std != 0:
            metric_result['z_score'] = (phi_mean - other_mean) / other_std
        else:
            metric_result['z_score'] = np.nan
            
        # Perform individual comparisons with each scaling factor
        individual_comparisons = {}
        for factor in scaling_factors:
            if factor != phi_value and factor in metric_data:
                factor_data = metric_data[factor]
                
                # T-test
                ind_t_stat, ind_p_value = stats.ttest_ind(phi_data, factor_data, equal_var=False)
                
                # Effect size
                ind_effect_size = calculate_cohens_d(phi_data, factor_data)
                
                individual_comparisons[factor] = {
                    'mean': np.mean(factor_data),
                    'std': np.std(factor_data, ddof=1),
                    't_statistic': ind_t_stat,
                    'p_value': ind_p_value,
                    'effect_size': ind_effect_size,
                    'significant': ind_p_value < alpha
                }
        
        metric_result['individual_comparisons'] = individual_comparisons
        
        # Perform non-parametric tests as well
        # Mann-Whitney U test
        u_stat, mw_p_value = stats.mannwhitneyu(phi_data, combined_other, alternative='two-sided')
        metric_result['mannwhitney_p_value'] = mw_p_value
        
        # Kolmogorov-Smirnov test (distribution comparison)
        ks_stat, ks_p_value = stats.ks_2samp(phi_data, combined_other)
        metric_result['ks_p_value'] = ks_p_value
        
        results[metric_name] = metric_result
    
    # Apply multiple testing correction
    if p_values:
        metric_names, p_vals = zip(*p_values)
        
        # Bonferroni correction
        bonferroni_p = np.minimum(np.array(p_vals) * len(p_vals), 1.0)
        
        # False Discovery Rate correction (Benjamini-Hochberg)
        reject, fdr_p, _, _ = multipletests(p_vals, alpha=alpha, method='fdr_bh')
        
        # Update results with corrected p-values
        for i, metric_name in enumerate(metric_names):
            results[metric_name]['bonferroni_p'] = bonferroni_p[i]
            results[metric_name]['fdr_p'] = fdr_p[i]
            results[metric_name]['significant_after_correction'] = {
                'bonferroni': bonferroni_p[i] < alpha,
                'fdr': fdr_p[i] < alpha
            }
    
    # Generate plots if requested
    if plot:
        if not output_dir:
            output_dir = 'plots'
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot comparing values at phi vs. other scaling factors
        if results:
            # Bar plot of means with error bars
            plt.figure(figsize=(10, 6))
            
            metrics_to_plot = list(results.keys())
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            phi_means = [results[m]['phi_mean'] for m in metrics_to_plot]
            phi_errors = [results[m]['phi_std'] / np.sqrt(len(metrics[m][phi_value])) for m in metrics_to_plot]
            
            other_means = [results[m]['other_mean'] for m in metrics_to_plot]
            other_errors = [results[m]['other_std'] / np.sqrt(len(metrics[m][phi_value])) for m in metrics_to_plot]
            
            plt.bar(x - width/2, phi_means, width, label=f'Phi ({phi_value})', yerr=phi_errors, capsize=5)
            plt.bar(x + width/2, other_means, width, label='Other Scaling Factors', yerr=other_errors, capsize=5)
            
            plt.xlabel('Metrics')
            plt.ylabel('Value')
            plt.title('Comparison of Metrics: Phi vs. Other Scaling Factors')
            plt.xticks(x, metrics_to_plot, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_path / "phi_vs_other_comparison.png", dpi=300)
            plt.close()
            
            # Plot of percent differences
            plt.figure(figsize=(10, 6))
            
            metrics_with_pct = [m for m in metrics_to_plot if not np.isnan(results[m].get('percent_difference', np.nan))]
            pct_diffs = [results[m]['percent_difference'] for m in metrics_with_pct]
            
            colors = ['green' if results[m]['significant'] else 'gray' for m in metrics_with_pct]
            
            bars = plt.bar(metrics_with_pct, pct_diffs, color=colors)
            
            # Add significance markers
            for i, m in enumerate(metrics_with_pct):
                if results[m]['significant']:
                    if results[m]['p_value'] < 0.001:
                        marker = '***'
                    elif results[m]['p_value'] < 0.01:
                        marker = '**'
                    else:
                        marker = '*'
                    
                    plt.text(i, pct_diffs[i] + 0.5, marker, ha='center')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Metrics')
            plt.ylabel('Percent Difference (%)')
            plt.title('Percent Difference of Phi Values vs. Other Scaling Factors')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(output_path / "phi_percent_difference.png", dpi=300)
            plt.close()
            
            # Plot p-values
            plt.figure(figsize=(10, 6))
            
            p_vals = [results[m]['p_value'] for m in metrics_to_plot]
            
            plt.bar(metrics_to_plot, p_vals, color=['green' if p < alpha else 'red' for p in p_vals])
            plt.axhline(y=alpha, color='red', linestyle='--', label=f'Significance Level ({alpha})')
            
            plt.xlabel('Metrics')
            plt.ylabel('p-value')
            plt.title('Statistical Significance of Phi vs. Other Scaling Factors')
            plt.xticks(rotation=45, ha='right')
            plt.yscale('log')  # Log scale for better visualization
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_path / "phi_significance_p_values.png", dpi=300)
            plt.close()
            
            # Detailed plots for each metric
            for metric_name, metric_result in results.items():
                if 'individual_comparisons' in metric_result:
                    # Plot comparing phi to each individual scaling factor
                    plt.figure(figsize=(12, 6))
                    
                    # Collect data
                    factors = sorted([f for f in scaling_factors if f != phi_value and f in metrics[metric_name]])
                    factor_means = [metrics[metric_name][f].mean() for f in factors]
                    factor_errors = [
                        metrics[metric_name][f].std() / np.sqrt(len(metrics[metric_name][f])) 
                        for f in factors
                    ]
                    
                    # Add phi at the end for emphasis
                    all_factors = factors + [phi_value]
                    all_means = factor_means + [metrics[metric_name][phi_value].mean()]
                    all_errors = factor_errors + [
                        metrics[metric_name][phi_value].std() / np.sqrt(len(metrics[metric_name][phi_value]))
                    ]
                    
                    # Set colors - highlight phi
                    colors = ['gray' for _ in factors] + ['green']
                    
                    # Plot
                    plt.errorbar(
                        all_factors, all_means, yerr=all_errors, 
                        fmt='o', capsize=5, ecolor='black', 
                        color='blue', markersize=8
                    )
                    
                    # Connect points with line
                    plt.plot(all_factors, all_means, 'b-', alpha=0.5)
                    
                    # Highlight phi
                    plt.plot(phi_value, all_means[-1], 'o', color='red', markersize=10)
                    
                    plt.xlabel('Scaling Factor')
                    plt.ylabel(metric_name)
                    plt.title(f'{metric_name} vs. Scaling Factor (highlighting phi)')
                    plt.grid(True, alpha=0.3)
                    
                    # Add vertical line at phi
                    plt.axvline(x=phi_value, color='red', linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(output_path / f"{metric_name}_phi_comparison.png", dpi=300)
                    plt.close()
    
    # Save results to CSV
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        for metric_name, metric_result in results.items():
            row = {
                'metric': metric_name,
                'phi_mean': metric_result['phi_mean'],
                'phi_std': metric_result['phi_std'],
                'other_mean': metric_result['other_mean'],
                'other_std': metric_result['other_std'],
                'difference': metric_result['phi_mean'] - metric_result['other_mean'],
                'percent_difference': metric_result.get('percent_difference', np.nan),
                'z_score': metric_result.get('z_score', np.nan),
                't_statistic': metric_result['t_statistic'],
                'p_value': metric_result['p_value'],
                'mannwhitney_p_value': metric_result.get('mannwhitney_p_value', np.nan),
                'ks_p_value': metric_result.get('ks_p_value', np.nan),
                'bonferroni_p': metric_result.get('bonferroni_p', np.nan),
                'fdr_p': metric_result.get('fdr_p', np.nan),
                'effect_size': metric_result['effect_size'],
                'significant': metric_result['significant'],
                'significant_after_bonferroni': metric_result.get('significant_after_correction', {}).get('bonferroni', False),
                'significant_after_fdr': metric_result.get('significant_after_correction', {}).get('fdr', False)
            }
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / "phi_significance_results.csv", index=False)
    
    return results

def calculate_cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two datasets.
    
    Parameters:
        data1 (np.ndarray): First dataset
        data2 (np.ndarray): Second dataset
    
    Returns:
        float: Cohen's d effect size
    """
    # Calculate means
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    # Calculate standard deviations
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)
    
    # Calculate pooled standard deviation
    n1 = len(data1)
    n2 = len(data2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d

if __name__ == "__main__":
    # Simple example usage
    # Define some synthetic data
    scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5, PHI, 2.0, 2.5, 3.0]
    
    # Create synthetic metrics that respond differently to phi
    np.random.seed(42)
    metrics = {
        'phi_sensitive': {
            factor: np.random.normal(
                0.5 if factor != PHI else 0.8, 0.1, size=20
            ) for factor in scaling_factors
        },
        'phi_insensitive': {
            factor: np.random.normal(
                0.5, 0.1, size=20
            ) for factor in scaling_factors
        },
        'phi_resonant': {
            factor: np.random.normal(
                0.3 + 0.7 * np.exp(-10 * (factor - PHI)**2), 0.1, size=20
            ) for factor in scaling_factors
        }
    }
    
    # Run analysis
    results = analyze_phi_significance(
        scaling_factors,
        metrics,
        phi_value=PHI,
        output_dir='data',
        plot=True
    )
    
    # Print results
    for metric_name, metric_result in results.items():
        print(f"\n{metric_name}:")
        print(f"  Phi mean: {metric_result['phi_mean']:.4f} ± {metric_result['phi_std']:.4f}")
        print(f"  Other mean: {metric_result['other_mean']:.4f} ± {metric_result['other_std']:.4f}")
        print(f"  Difference: {metric_result['phi_mean'] - metric_result['other_mean']:.4f}")
        if 'percent_difference' in metric_result:
            print(f"  Percent difference: {metric_result['percent_difference']:.2f}%")
        if 'z_score' in metric_result:
            print(f"  Z-score: {metric_result['z_score']:.2f}")
        print(f"  p-value: {metric_result['p_value']:.4f}")
        print(f"  Effect size (Cohen's d): {metric_result['effect_size']:.4f}")
        print(f"  Significant: {metric_result['significant']}")
        if 'significant_after_correction' in metric_result:
            print(f"  Significant after Bonferroni: {metric_result['significant_after_correction']['bonferroni']}")
            print(f"  Significant after FDR: {metric_result['significant_after_correction']['fdr']}")
=======
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
>>>>>>> 67621917d847af621febdd13bfc67b86a99b6e65

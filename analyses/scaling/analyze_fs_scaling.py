<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing scaling properties in the RGQS system.

This module implements analyses of how different scaling properties affect
quantum evolution and metrics, with a particular focus on the
scaling factor f_s.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

def analyze_fs_scaling(
    scaling_factors: List[float],
    metrics: Dict[str, Dict[float, np.ndarray]],
    output_dir: Optional[str] = None,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Analyze how quantum metrics scale with different scaling factors.
    
    Parameters:
        scaling_factors (List[float]): List of scaling factors used
        metrics (Dict[str, Dict[float, np.ndarray]]): Dictionary mapping metric names to
                                                     nested dictionaries that map scaling factors
                                                     to data arrays
        output_dir (Optional[str]): Directory to save results
        plot (bool): Whether to generate plots
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    results = {}
    
    # Calculate mean and std for each metric across scaling factors
    for metric_name, metric_data in metrics.items():
        metric_results = {
            'means': {},
            'stds': {},
            'scaling_trend': None,
            'correlation': None
        }
        
        means = []
        for factor in scaling_factors:
            if factor in metric_data:
                data = metric_data[factor]
                metric_results['means'][factor] = np.mean(data)
                metric_results['stds'][factor] = np.std(data, ddof=1)
                means.append((factor, np.mean(data)))
        
        # Convert to arrays for analysis
        if means:
            factors, values = zip(*means)
            factors = np.array(factors)
            values = np.array(values)
            
            # Calculate correlation between scaling factor and metric
            if len(factors) > 2:  # Need at least 3 points for meaningful correlation
                correlation = np.corrcoef(factors, values)[0, 1]
                metric_results['correlation'] = correlation
                
                # Try to fit linear and power-law models
                try:
                    # Linear fit: y = ax + b
                    linear_fit = np.polyfit(factors, values, 1)
                    linear_r2 = calculate_r2(factors, values, np.poly1d(linear_fit))
                    
                    # Power-law fit: y = a * x^b
                    # Use log-log fit: log(y) = log(a) + b*log(x)
                    # Only use positive values for log
                    pos_mask = (factors > 0) & (values > 0)
                    if np.sum(pos_mask) > 2:
                        log_factors = np.log(factors[pos_mask])
                        log_values = np.log(values[pos_mask])
                        power_fit = np.polyfit(log_factors, log_values, 1)
                        power_law_r2 = calculate_r2(
                            log_factors, 
                            log_values, 
                            lambda x: power_fit[0] * x + power_fit[1]
                        )
                        
                        # Determine better fit
                        if power_law_r2 > linear_r2:
                            a, b = np.exp(power_fit[1]), power_fit[0]
                            metric_results['scaling_trend'] = {
                                'type': 'power-law',
                                'equation': f"y = {a:.4f} * x^{b:.4f}",
                                'parameters': {'a': a, 'b': b},
                                'r2': power_law_r2
                            }
                        else:
                            a, b = linear_fit
                            metric_results['scaling_trend'] = {
                                'type': 'linear',
                                'equation': f"y = {a:.4f}x + {b:.4f}",
                                'parameters': {'a': a, 'b': b},
                                'r2': linear_r2
                            }
                    else:
                        a, b = linear_fit
                        metric_results['scaling_trend'] = {
                            'type': 'linear',
                            'equation': f"y = {a:.4f}x + {b:.4f}",
                            'parameters': {'a': a, 'b': b},
                            'r2': linear_r2
                        }
                except Exception as e:
                    metric_results['fitting_error'] = str(e)
        
        results[metric_name] = metric_results
        
        # Generate plots if requested
        if plot:
            if not output_dir:
                output_dir = 'plots'
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if means:
                plt.figure(figsize=(10, 6))
                x = np.array(factors)
                y = np.array(values)
                yerr = np.array([metric_results['stds'].get(f, 0) for f in factors])
                
                plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label=metric_name)
                
                # Add trend line if available
                try:
                    if metric_results.get('scaling_trend') and isinstance(metric_results['scaling_trend'], dict):
                        trend = metric_results['scaling_trend']
                        x_smooth = np.linspace(min(x), max(x), 100)
                        
                        # Try to extract trend type
                        trend_type = trend.get('type')
                        parameters = trend.get('parameters', {})
                        
                        # Linear trend
                        if trend_type == 'linear' and isinstance(parameters, dict):
                            a = parameters.get('a')
                            b = parameters.get('b')
                            if a is not None and b is not None:
                                y_smooth = a * x_smooth + b
                                r2 = trend.get('r2', 0)
                                eq_str = trend.get('equation', f"y = {a:.4f}x + {b:.4f}")
                                plt.plot(x_smooth, y_smooth, 'r--', 
                                        label=f"Fit: {eq_str} (R² = {r2:.3f})")
                        
                        # Power-law trend
                        elif trend_type == 'power-law' and isinstance(parameters, dict):
                            a = parameters.get('a')
                            b = parameters.get('b')
                            if a is not None and b is not None:
                                y_smooth = a * x_smooth ** b
                                r2 = trend.get('r2', 0)
                                eq_str = trend.get('equation', f"y = {a:.4f} * x^{b:.4f}")
                                plt.plot(x_smooth, y_smooth, 'r--', 
                                        label=f"Fit: {eq_str} (R² = {r2:.3f})")
                except Exception as e:
                    # Log error but continue execution
                    print(f"Warning: Error plotting trend line for {metric_name}: {str(e)}")
                
                plt.xlabel('Scaling Factor (f_s)')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} vs. Scaling Factor')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(output_path / f"{metric_name}_vs_scaling.png", dpi=300)
                plt.close()
    
    # Generate combined plot if multiple metrics
    if plot and len(metrics) > 1:
        plt.figure(figsize=(12, 8))
        
        for metric_name, metric_results in results.items():
            if 'means' in metric_results:
                factors = list(metric_results['means'].keys())
                values = list(metric_results['means'].values())
                
                if factors and values:
                    # Normalize to [0, 1] for comparison
                    min_val = min(values)
                    max_val = max(values)
                    if max_val > min_val:  # Avoid division by zero
                        norm_values = [(v - min_val) / (max_val - min_val) for v in values]
                        plt.plot(factors, norm_values, 'o-', label=metric_name)
        
        plt.xlabel('Scaling Factor (f_s)')
        plt.ylabel('Normalized Metric Value')
        plt.title('Comparison of Metrics vs. Scaling Factor')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(output_path / "combined_metrics_vs_scaling.png", dpi=300)
        plt.close()
    
    # Save results to CSV
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        for factor in scaling_factors:
            row = {'scaling_factor': factor}
            
            for metric_name, metric_results in results.items():
                if factor in metric_results['means']:
                    row[f"{metric_name}_mean"] = metric_results['means'][factor]
                    row[f"{metric_name}_std"] = metric_results['stds'][factor]
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / "fs_scaling_results.csv", index=False)
    
    return results

def calculate_r2(x: np.ndarray, y: np.ndarray, model: callable) -> float:
    """
    Calculate the coefficient of determination (R^2) for a model.
    
    Parameters:
        x (np.ndarray): Input data
        y (np.ndarray): Observed values
        model (callable): Model function that takes x and returns predicted y
    
    Returns:
        float: R^2 value
    """
    y_pred = model(x)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    
    # Handle edge case where all y values are identical
    if ss_total == 0:
        # Model is perfect if predictions match exactly, otherwise it's the worst possible
        return 1.0 if np.allclose(y, y_pred) else 0.0
    
    r2 = 1 - (ss_residual / ss_total)
    
    # R^2 can be negative if model is worse than horizontal line
    # Clip to [0, 1] for better interpretability
    return max(0, min(1, r2))

if __name__ == "__main__":
    # Simple example usage
    # Define some synthetic data
    scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 1.618, 2.0, 2.5, 3.0]
    
    # Create synthetic metrics that scale differently with f_s
    np.random.seed(42)
    metrics = {
        'linear_metric': {
            factor: 0.5 * factor + 0.2 + 0.1 * np.random.randn(10)
            for factor in scaling_factors
        },
        'quadratic_metric': {
            factor: 0.2 * factor**2 + 0.1 + 0.15 * np.random.randn(10)
            for factor in scaling_factors
        },
        'phi_resonant_metric': {
            factor: 0.3 + 0.7 * np.exp(-10 * (factor - 1.618)**2) + 0.1 * np.random.randn(10)
            for factor in scaling_factors
        }
    }
    
    # Run analysis
    results = analyze_fs_scaling(
        scaling_factors,
        metrics,
        output_dir='data',
        plot=True
    )
    
    # Print results
    for metric_name, metric_results in results.items():
        print(f"\n{metric_name}:")
        
        if 'correlation' in metric_results:
            print(f"  Correlation with scaling factor: {metric_results['correlation']:.4f}")
            
        if 'scaling_trend' in metric_results and metric_results['scaling_trend'] and isinstance(metric_results['scaling_trend'], dict):
            trend = metric_results['scaling_trend']
            if 'equation' in trend and 'r2' in trend:
                print(f"  Best fit: {trend['equation']} (R² = {trend['r2']:.3f})")
            elif 'type' in trend and 'parameters' in trend:
                ttype = trend['type']
                params = trend['parameters']
                if ttype == 'linear' and 'a' in params and 'b' in params:
                    a, b = params['a'], params['b']
                    r2 = trend.get('r2', 0)
                    print(f"  Best fit: y = {a:.4f}x + {b:.4f} (R² = {r2:.3f})")
                elif ttype == 'power-law' and 'a' in params and 'b' in params:
                    a, b = params['a'], params['b']
                    r2 = trend.get('r2', 0)
                    print(f"  Best fit: y = {a:.4f} * x^{b:.4f} (R² = {r2:.3f})")
=======
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
>>>>>>> 67621917d847af621febdd13bfc67b86a99b6e65

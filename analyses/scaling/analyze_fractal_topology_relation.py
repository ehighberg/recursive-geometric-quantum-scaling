<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for analyzing the relationship between fractal dimensions and topological invariants.

This module implements analyses of how fractal dimensions relate to topological properties
in quantum systems, with a focus on identifying phase transitions and invariant structures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

def analyze_fractal_topology_relation(
    fractal_dimensions: Dict[float, np.ndarray],
    topological_invariants: Dict[float, Dict[str, np.ndarray]],
    scaling_factors: List[float],
    invariant_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    plot: bool = True,
    smoothing: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze the relationship between fractal dimensions and topological invariants.
    
    Parameters:
        fractal_dimensions (Dict[float, np.ndarray]): Dictionary mapping scaling factors
                                                     to arrays of fractal dimensions
        topological_invariants (Dict[float, Dict[str, np.ndarray]]): Dictionary mapping scaling factors
                                                                    to dictionaries that map invariant names
                                                                    to arrays of invariant values
        scaling_factors (List[float]): List of scaling factors used
        invariant_names (Optional[List[str]]): List of invariant names to analyze
                                               (default: all available invariants)
        output_dir (Optional[str]): Directory to save results
        plot (bool): Whether to generate plots
        smoothing (float): Smoothing factor for phase diagram (0 for no smoothing)
    
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    results = {}
    
    # If invariant names are not provided, use all available invariants
    if invariant_names is None:
        # Find first scaling factor with topological invariants
        for factor in scaling_factors:
            if factor in topological_invariants:
                invariant_names = list(topological_invariants[factor].keys())
                break
        
        if invariant_names is None:
            raise ValueError("No topological invariants found")
    
    # Compute correlations between fractal dimensions and topological invariants
    correlations = {}
    for inv_name in invariant_names:
        # Collect paired data for correlation analysis
        paired_data = []
        for factor in scaling_factors:
            if (factor in fractal_dimensions and 
                factor in topological_invariants and 
                inv_name in topological_invariants[factor]):
                
                fd = fractal_dimensions[factor]
                inv = topological_invariants[factor][inv_name]
                
                # Ensure arrays have same length (if not, use the minimum length)
                min_len = min(len(fd), len(inv))
                paired_data.append((factor, fd[:min_len], inv[:min_len]))
        
        if not paired_data:
            continue
            
        # Compute correlation for each scaling factor
        factor_correlations = {}
        overall_fd = []
        overall_inv = []
        
        for factor, fd, inv in paired_data:
            corr, p_value = stats.pearsonr(fd, inv)
            factor_correlations[factor] = {
                'correlation': corr,
                'p_value': p_value,
                'num_samples': len(fd)
            }
            
            # Append data for overall correlation
            overall_fd.extend(fd)
            overall_inv.extend(inv)
        
        # Compute overall correlation
        overall_corr, overall_p = stats.pearsonr(overall_fd, overall_inv)
        
        correlations[inv_name] = {
            'factor_correlations': factor_correlations,
            'overall_correlation': overall_corr,
            'overall_p_value': overall_p,
            'num_samples': len(overall_fd)
        }
    
    results['correlations'] = correlations
    
    # Create phase diagram data
    phase_diagrams = {}
    for inv_name in invariant_names:
        # Collect data for phase diagram
        fd_values = []
        factor_values = []
        inv_values = []
        
        for factor in scaling_factors:
            if (factor in fractal_dimensions and 
                factor in topological_invariants and 
                inv_name in topological_invariants[factor]):
                
                fd = fractal_dimensions[factor]
                inv = topological_invariants[factor][inv_name]
                
                # Ensure arrays have same length
                min_len = min(len(fd), len(inv))
                
                fd_values.extend(fd[:min_len])
                factor_values.extend([factor] * min_len)
                inv_values.extend(inv[:min_len])
        
        if not fd_values:
            continue
            
        # Create grid for interpolation
        grid_size = 100
        fd_grid = np.linspace(min(fd_values), max(fd_values), grid_size)
        factor_grid = np.linspace(min(factor_values), max(factor_values), grid_size)
        
        # Create meshgrid
        fd_mesh, factor_mesh = np.meshgrid(fd_grid, factor_grid)
        
        # Interpolate invariant values onto grid
        points = np.column_stack((fd_values, factor_values))
        inv_grid = griddata(points, inv_values, (fd_mesh, factor_mesh), method='linear')
        
        # Apply smoothing if requested
        if smoothing > 0:
            inv_grid = gaussian_filter(inv_grid, sigma=smoothing)
        
        phase_diagrams[inv_name] = {
            'fd_mesh': fd_mesh,
            'factor_mesh': factor_mesh,
            'inv_grid': inv_grid
        }
    
    results['phase_diagrams'] = phase_diagrams
    
    # Identify phase transitions and critical points
    phase_transitions = {}
    for inv_name, phase_diagram in phase_diagrams.items():
        inv_grid = phase_diagram['inv_grid']
        fd_grid = phase_diagram['fd_mesh'][0, :]
        factor_grid = phase_diagram['factor_mesh'][:, 0]
        
        # Calculate gradient magnitude of invariant field
        grad_y, grad_x = np.gradient(inv_grid)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify potential phase transitions (high gradient magnitude)
        threshold = np.percentile(gradient_magnitude[~np.isnan(gradient_magnitude)], 90)
        transitions = gradient_magnitude > threshold
        
        # Find critical points (local maxima of gradient magnitude)
        # Use a simple peak finding approach for demonstration
        critical_points = []
        for i in range(1, gradient_magnitude.shape[0]-1):
            for j in range(1, gradient_magnitude.shape[1]-1):
                if not np.isnan(gradient_magnitude[i, j]):
                    neighborhood = gradient_magnitude[i-1:i+2, j-1:j+2]
                    if np.nanmax(neighborhood) == gradient_magnitude[i, j]:
                        critical_points.append({
                            'fd': fd_grid[j],
                            'scaling_factor': factor_grid[i],
                            'invariant_value': inv_grid[i, j],
                            'gradient_magnitude': gradient_magnitude[i, j]
                        })
        
        # Sort critical points by gradient magnitude
        critical_points.sort(key=lambda p: p['gradient_magnitude'], reverse=True)
        
        # Take top 5 critical points or fewer if less exist
        top_critical_points = critical_points[:min(5, len(critical_points))]
        
        phase_transitions[inv_name] = {
            'gradient_magnitude': gradient_magnitude,
            'transition_mask': transitions,
            'critical_points': top_critical_points
        }
    
    results['phase_transitions'] = phase_transitions
    
    # Generate plots if requested
    if plot:
        if not output_dir:
            output_dir = 'plots'
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot correlation matrix
        if correlations:
            plt.figure(figsize=(10, 8))
            
            # Create correlation matrix
            inv_list = list(correlations.keys())
            corr_matrix = np.zeros((len(inv_list), len(scaling_factors)))
            
            for i, inv_name in enumerate(inv_list):
                if inv_name in correlations:
                    for j, factor in enumerate(scaling_factors):
                        if (factor in correlations[inv_name]['factor_correlations']):
                            corr_matrix[i, j] = correlations[inv_name]['factor_correlations'][factor]['correlation']
            
            # Plot heatmap
            plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            
            plt.xlabel('Scaling Factor')
            plt.ylabel('Topological Invariant')
            plt.title('Correlation Between Fractal Dimensions and Topological Invariants')
            
            # Set tick labels
            plt.yticks(np.arange(len(inv_list)), inv_list)
            plt.xticks(np.arange(len(scaling_factors)), [f"{f:.2f}" for f in scaling_factors], rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / "fractal_topology_correlation.png", dpi=300)
            plt.close()
            
            # Plot overall correlations as bar chart
            plt.figure(figsize=(10, 6))
            
            inv_list = list(correlations.keys())
            overall_corrs = [correlations[inv]['overall_correlation'] for inv in inv_list]
            
            bars = plt.bar(inv_list, overall_corrs)
            
            # Add significance markers
            for i, inv in enumerate(inv_list):
                if correlations[inv]['overall_p_value'] < 0.001:
                    marker = '***'
                elif correlations[inv]['overall_p_value'] < 0.01:
                    marker = '**'
                elif correlations[inv]['overall_p_value'] < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    plt.text(i, overall_corrs[i] + 0.05 * np.sign(overall_corrs[i]), 
                             marker, ha='center')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Topological Invariant')
            plt.ylabel('Correlation with Fractal Dimension')
            plt.title('Overall Correlation Between Fractal Dimensions and Topological Invariants')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(-1.1, 1.1)
            plt.tight_layout()
            
            plt.savefig(output_path / "fractal_topology_overall_correlation.png", dpi=300)
            plt.close()
        
        # Plot phase diagrams
        for inv_name, phase_diagram in phase_diagrams.items():
            plt.figure(figsize=(10, 8))
            
            fd_mesh = phase_diagram['fd_mesh']
            factor_mesh = phase_diagram['factor_mesh']
            inv_grid = phase_diagram['inv_grid']
            
            # Plot phase diagram as a contour plot
            contour = plt.contourf(fd_mesh, factor_mesh, inv_grid, 20, cmap='viridis')
            plt.colorbar(contour, label=f'{inv_name} Value')
            
            # Add phase transitions if available
            if inv_name in phase_transitions:
                # Plot transition regions as contour
                transitions = phase_transitions[inv_name]['transition_mask']
                gradient_mag = phase_transitions[inv_name]['gradient_magnitude']
                
                # Plot gradient magnitude contour
                plt.contour(fd_mesh, factor_mesh, gradient_mag, 
                           levels=np.linspace(np.nanmin(gradient_mag), np.nanmax(gradient_mag), 5),
                           colors='white', alpha=0.5, linewidths=1)
                
                # Mark critical points
                for point in phase_transitions[inv_name]['critical_points']:
                    plt.plot(point['fd'], point['scaling_factor'], 'r*', markersize=10)
            
            plt.xlabel('Fractal Dimension')
            plt.ylabel('Scaling Factor')
            plt.title(f'Phase Diagram: {inv_name} vs. Fractal Dimension and Scaling Factor')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path / f"fractal_topology_phase_diagram_{inv_name}.png", dpi=300)
            plt.close()
            
            # Plot combined phase diagram showing critical points
            if inv_name in phase_transitions:
                plt.figure(figsize=(10, 8))
                
                # Plot invariant value as background
                plt.contourf(fd_mesh, factor_mesh, inv_grid, 20, cmap='viridis', alpha=0.7)
                plt.colorbar(label=f'{inv_name} Value')
                
                # Overlay gradient magnitude
                gradient_mag = phase_transitions[inv_name]['gradient_magnitude']
                gradient_norm = gradient_mag / np.nanmax(gradient_mag)
                
                # Use a different colormap for gradient
                plt.contour(fd_mesh, factor_mesh, gradient_norm, 
                           levels=np.linspace(0.5, 1.0, 5),
                           colors='white', alpha=0.8, linewidths=1)
                
                # Mark critical points
                for point in phase_transitions[inv_name]['critical_points']:
                    plt.plot(point['fd'], point['scaling_factor'], 'r*', markersize=10,
                            label=f"Critical Point: ({point['fd']:.2f}, {point['scaling_factor']:.2f})")
                
                # Add golden ratio line if in range
                phi = (1 + np.sqrt(5)) / 2
                if min(scaling_factors) <= phi <= max(scaling_factors):
                    plt.axhline(y=phi, color='yellow', linestyle='--', 
                               label=f'Golden Ratio (Φ ≈ {phi:.3f})')
                
                plt.xlabel('Fractal Dimension')
                plt.ylabel('Scaling Factor')
                plt.title(f'Phase Transitions in {inv_name} vs. Fractal Dimension and Scaling Factor')
                plt.grid(True, alpha=0.3)
                
                # Add legend for the first few critical points only
                handles, labels = plt.gca().get_legend_handles_labels()
                if len(handles) > 6:  # Limit to 5 critical points + phi line
                    handles = handles[:6]
                    labels = labels[:6]
                plt.legend(handles, labels, loc='best')
                
                plt.tight_layout()
                plt.savefig(output_path / f"fractal_topology_phase_transitions_{inv_name}.png", dpi=300)
                plt.close()
    
    # Save results to CSV
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save correlations
        correlation_data = []
        for inv_name, corr_info in correlations.items():
            for factor, factor_corr in corr_info['factor_correlations'].items():
                row = {
                    'invariant': inv_name,
                    'scaling_factor': factor,
                    'correlation': factor_corr['correlation'],
                    'p_value': factor_corr['p_value'],
                    'num_samples': factor_corr['num_samples']
                }
                correlation_data.append(row)
        
        # Add overall correlations
        for inv_name, corr_info in correlations.items():
            row = {
                'invariant': inv_name,
                'scaling_factor': 'overall',
                'correlation': corr_info['overall_correlation'],
                'p_value': corr_info['overall_p_value'],
                'num_samples': corr_info['num_samples']
            }
            correlation_data.append(row)
        
        correlation_df = pd.DataFrame(correlation_data)
        correlation_df.to_csv(output_path / "fractal_topology_correlation.csv", index=False)
        
        # Save critical points
        critical_point_data = []
        for inv_name, transition_info in phase_transitions.items():
            for point in transition_info['critical_points']:
                row = {
                    'invariant': inv_name,
                    'fractal_dimension': point['fd'],
                    'scaling_factor': point['scaling_factor'],
                    'invariant_value': point['invariant_value'],
                    'gradient_magnitude': point['gradient_magnitude']
                }
                critical_point_data.append(row)
        
        critical_point_df = pd.DataFrame(critical_point_data)
        critical_point_df.to_csv(output_path / "fractal_topology_critical_points.csv", index=False)
    
    return results

if __name__ == "__main__":
    # Simple example usage
    # Define some synthetic data
    scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5, (1 + np.sqrt(5)) / 2, 2.0, 2.5, 3.0]
    
    # Generate synthetic fractal dimensions
    np.random.seed(42)
    fractal_dimensions = {
        factor: 1.2 + 0.3 * np.sin(factor * np.pi) + 0.05 * np.random.randn(10)
        for factor in scaling_factors
    }
    
    # Generate synthetic topological invariants
    topological_invariants = {
        factor: {
            'chern_number': np.round(0.5 + 0.5 * np.tanh((factor - 1.7) * 2) + 0.2 * np.random.randn(10)),
            'winding_number': np.round(factor / 2) + 0.1 * np.random.randn(10),
            'z2_index': np.round(np.abs(np.sin(factor * np.pi))) + 0.05 * np.random.randn(10)
        }
        for factor in scaling_factors
    }
    
    # Run analysis
    results = analyze_fractal_topology_relation(
        fractal_dimensions,
        topological_invariants,
        scaling_factors,
        output_dir='data',
        plot=True
    )
    
    # Print results
    print("\nCorrelations between fractal dimensions and topological invariants:")
    for inv_name, corr_info in results['correlations'].items():
        print(f"\n{inv_name}:")
        print(f"  Overall correlation: {corr_info['overall_correlation']:.4f} "
              f"(p-value: {corr_info['overall_p_value']:.4f})")
        
        print("  By scaling factor:")
        for factor, factor_corr in sorted(corr_info['factor_correlations'].items()):
            print(f"    {factor:.3f}: {factor_corr['correlation']:.4f} "
                  f"(p-value: {factor_corr['p_value']:.4f})")
    
    print("\nIdentified phase transitions and critical points:")
    for inv_name, transition_info in results['phase_transitions'].items():
        print(f"\n{inv_name}:")
        
        if transition_info['critical_points']:
            print("  Critical points:")
            for i, point in enumerate(transition_info['critical_points']):
                print(f"    {i+1}. Fractal dimension: {point['fd']:.3f}, "
                      f"Scaling factor: {point['scaling_factor']:.3f}")
        else:
            print("  No critical points identified.")
=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the relationship between fractal properties and topological invariants.

This script explores how fractal dimensions and topological invariants are correlated
across different scaling factors, with a focus on identifying patterns and critical points.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from qutip import Qobj, tensor, sigmaz, sigmax, qeye, basis
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

# Import simulation components
from simulations.scripts.evolve_circuit import run_phi_scaled_twoqubit_circuit
from analyses.fractal_analysis import compute_energy_spectrum, estimate_fractal_dimension
from analyses.topological_invariants import compute_phi_sensitive_winding, compute_phi_sensitive_z2, compute_phi_resonant_berry_phase
from constants import PHI

def analyze_fractal_topology_relation(fs_values=None, save_results=True):
    """
    Analyze the relationship between fractal properties and topological invariants
    across different scaling factors.
    
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
        # Create a denser grid for better correlation analysis
        fs_values = np.linspace(0.5, 3.0, 51)
        # Add PHI and some Fibonacci ratios
        phi_approx = [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]
        fs_values = np.append(fs_values, [PHI] + [ratio[0]/ratio[1] for ratio in phi_approx])
        fs_values = np.sort(np.unique(fs_values))
    
    # Initialize results storage
    results = {
        'fs_values': fs_values,
        'band_gaps': np.zeros_like(fs_values),
        'fractal_dimensions': np.zeros_like(fs_values),
        'topological_invariants': np.zeros_like(fs_values),
        'correlation_lengths': np.zeros_like(fs_values),
        'z2_indices': np.zeros_like(fs_values),
        'self_similarity_metrics': np.zeros_like(fs_values)
,
        'berry_phases': np.zeros_like(fs_values),
        'energy_spectra': []
    }
    
    print(f"Analyzing {len(fs_values)} f_s values")
    
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
        
        # Compute topological invariants using mathematically correct implementations
        # without artificial phi-based modifications
        winding = compute_phi_sensitive_winding(eigenstates, k_points, fs)
        
        # Compute Z2 index using the fixed implementation
        z2_index = compute_phi_sensitive_z2(eigenstates, k_points, fs)
        
        # Calculate Berry phase using proper mathematical definition
        berry_phase = compute_phi_resonant_berry_phase(eigenstates, fs)
        
        # Estimate correlation length
        if band_gap > 1e-10:
            correlation_length = 1.0 / band_gap
        else:
            correlation_length = np.nan
        
        # Calculate self-similarity metric from spectrum analysis
        if 'self_similar_regions' in spectrum_analysis:
            self_similarity = len(spectrum_analysis['self_similar_regions'])
        else:
            self_similarity = 0
        
        # Store results
        results['band_gaps'][i] = band_gap
        results['fractal_dimensions'][i] = fractal_dim
        results['topological_invariants'][i] = winding
        results['correlation_lengths'][i] = correlation_length
        results['z2_indices'][i] = z2_index
        results['berry_phases'][i] = berry_phase
        results['self_similarity_metrics'][i] = self_similarity
    
    # Calculate correlations between fractal and topological properties
    valid_indices = ~np.isnan(results['fractal_dimensions']) & ~np.isnan(results['topological_invariants'])
    
    if np.sum(valid_indices) > 2:  # Need at least 3 points for correlation
        fractal_topo_pearson, p_value_pearson = pearsonr(
            results['fractal_dimensions'][valid_indices],
            results['topological_invariants'][valid_indices]
        )
        
        fractal_topo_spearman, p_value_spearman = spearmanr(
            results['fractal_dimensions'][valid_indices],
            results['topological_invariants'][valid_indices]
        )
    else:
        fractal_topo_pearson, p_value_pearson = np.nan, np.nan
        fractal_topo_spearman, p_value_spearman = np.nan, np.nan
    
    # Store correlation results
    results['correlations'] = {
        'fractal_topo_pearson': fractal_topo_pearson,
        'p_value_pearson': p_value_pearson,
        'fractal_topo_spearman': fractal_topo_spearman,
        'p_value_spearman': p_value_spearman
    }
    
    # Convert results to DataFrame for easier handling
    df = pd.DataFrame({
        'f_s': results['fs_values'],
        'Band Gap': results['band_gaps'],
        'Fractal Dimension': results['fractal_dimensions'],
        'Topological Invariant': results['topological_invariants'],
        'Z2 Index': results['z2_indices'],
        'Correlation Length': results['correlation_lengths'],
 
        'Berry Phase': results['berry_phases'],
        'Self-Similarity': results['self_similarity_metrics']
    })
    
    # Add column indicating if value is phi
    df['Is Phi'] = np.isclose(df['f_s'], PHI, rtol=1e-10)
    
    # Print results table
    print("\nResults Table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Print correlation results
    print("\nCorrelation Analysis:")
    print(f"Pearson correlation (fractal dim vs topo inv): {fractal_topo_pearson:.4f} (p={p_value_pearson:.4f})")
    print(f"Spearman correlation (fractal dim vs topo inv): {fractal_topo_spearman:.4f} (p={p_value_spearman:.4f})")
    
    if save_results:
        # Save results to CSV
        output_dir = Path(".")
        df.to_csv(output_dir / "fractal_topology_relation.csv", index=False)
        print(f"Results saved to fractal_topology_relation.csv")
        
        # Create visualization
        create_fractal_topology_plots(results, output_dir)
    
    return results

def create_fractal_topology_plots(results, output_dir=None):
    """
    Create plots showing the relationship between fractal properties and topological invariants.
    
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
    
    # Plot fractal dimension vs topological invariant
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        results['fractal_dimensions'],
        results['topological_invariants'],
        c=results['fs_values'],
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Scale Factor (f_s)')
    
    # Highlight phi point
    phi_idx = np.argmin(np.abs(results['fs_values'] - PHI))
    ax1.scatter(
        results['fractal_dimensions'][phi_idx],
        results['topological_invariants'][phi_idx],
        s=100,
        facecolors='none',
        edgecolors='r',
        linewidth=2,
        label=f'φ ≈ {PHI:.4f}'
    )
    
    ax1.set_xlabel('Fractal Dimension')
    ax1.set_ylabel('Topological Invariant')
    ax1.set_title('Fractal Dimension vs. Topological Invariant')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add correlation information
    if 'correlations' in results:
        corr = results['correlations']
        if not np.isnan(corr['fractal_topo_pearson']):
            ax1.annotate(
                f"Pearson r = {corr['fractal_topo_pearson']:.4f} (p={corr['p_value_pearson']:.4f})\n"
                f"Spearman ρ = {corr['fractal_topo_spearman']:.4f} (p={corr['p_value_spearman']:.4f})",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                ha='left',
                va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8)
            )
    
    # Plot fractal dimension and topological invariant vs f_s
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2, label='Fractal Dimension')
    ax2.set_xlabel('Scale Factor (f_s)')
    ax2.set_ylabel('Fractal Dimension', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(results['fs_values'], results['topological_invariants'], 's-', color='#2ca02c', linewidth=2, label='Topological Invariant')
    ax2_twin.set_ylabel('Topological Invariant', color='#2ca02c')
    ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
    
    # Add vertical line at phi
    ax2.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.4f}')
    
    # Create combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_title('Fractal Dimension and Topological Invariant vs. Scale Factor')
    ax2.grid(True, alpha=0.3)
    
    # Plot self-similarity metric vs f_s
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(results['fs_values'], results['self_similarity_metrics'], 'o-', color='#d62728', linewidth=2)
    ax3.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.4f}')
    ax3.set_xlabel('Scale Factor (f_s)')
    ax3.set_ylabel('Self-Similarity Metric')
    ax3.set_title('Self-Similarity vs. Scale Factor')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot Z2 index vs f_s
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['fs_values'], results['z2_indices'], 'o-', color='#9467bd', linewidth=2, label='Z2 Index')
    ax4.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.4f}')
    ax4.set_xlabel('Scale Factor (f_s)')
    ax4.set_ylabel('Z2 Index')
    
    # Add Berry phase on secondary y-axis
    ax4_twin = ax4.twinx()
    ax4_twin.plot(results['fs_values'], results['berry_phases']/(2*np.pi), 's--', color='#8c564b', linewidth=1.5, label='Berry Phase/(2π)')
    ax4_twin.set_ylabel('Berry Phase/(2π)', color='#8c564b')
    ax4_twin.tick_params(axis='y', labelcolor='#8c564b')
    ax4.grid(True, alpha=0.3)
    
    # Create combined legend for Z2 and Berry phase
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines4_twin, labels4_twin = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines4 + lines4_twin, labels4 + labels4_twin, loc='upper right')
    ax4.set_title('Z2 Index and Berry Phase vs. Scale Factor')
    
    # Add overall title
    fig.suptitle('Relationship Between Fractal Properties and Topological Invariants', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / "fractal_topology_relation.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to fractal_topology_relation.png")
    
    # Create phase diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use fractal dimension and topological invariant as coordinates
    x = results['fractal_dimensions']
    y = results['topological_invariants']
    
    # Filter out NaN values
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    fs_valid = results['fs_values'][valid_indices]
    
    if len(x_valid) > 3:  # Need at least 4 points for clustering
        # Try to identify clusters/phases
        try:
            # Combine features for clustering
            features = np.column_stack([x_valid, y_valid])
            
            # Determine optimal number of clusters (2-4)
            inertias = []
            for k in range(1, 5):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (simplified)
            k_opt = 2  # Default
            for i in range(1, len(inertias)-1):
                if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) > 2:
                    k_opt = i + 1
                    break
            
            # Perform clustering with optimal k
            kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Plot clusters
            scatter = ax.scatter(x_valid, y_valid, c=clusters, cmap='tab10', s=50, alpha=0.7)
            
            # Add cluster centers
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='X')
            
            # Add legend for clusters
            legend1 = ax.legend(*scatter.legend_elements(),
                                title="Phases")
            ax.add_artist(legend1)
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            # Fallback to simple scatter plot
            scatter = ax.scatter(x_valid, y_valid, c=fs_valid, cmap='viridis', s=50, alpha=0.7)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Scale Factor (f_s)')
    else:
        # Fallback to simple scatter plot
        scatter = ax.scatter(x_valid, y_valid, c=fs_valid, cmap='viridis', s=50, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Scale Factor (f_s)')
    
    # Highlight phi point if it's valid
    phi_idx = np.argmin(np.abs(results['fs_values'] - PHI))
    if not np.isnan(x[phi_idx]) and not np.isnan(y[phi_idx]):
        ax.scatter(
            x[phi_idx],
            y[phi_idx],
            s=100,
            facecolors='none',
            edgecolors='r',
            linewidth=2,
            label=f'φ ≈ {PHI:.4f}'
        )
    
    ax.set_xlabel('Fractal Dimension')
    ax.set_ylabel('Topological Invariant')
    ax.set_title('Phase Diagram: Fractal Dimension vs. Topological Invariant')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save phase diagram
    plt.savefig(output_dir / "fractal_topology_phase_diagram.png", dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved to fractal_topology_phase_diagram.png")
    
    plt.close('all')

if __name__ == "__main__":
    # Run analysis
    analyze_fractal_topology_relation()
>>>>>>> 67621917d847af621febdd13bfc67b86a99b6e65

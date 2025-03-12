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
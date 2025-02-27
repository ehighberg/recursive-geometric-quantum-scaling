#!/usr/bin/env python
# run_phi_resonant_analysis.py

"""
Run phi-resonant analysis to compare standard and phi-recursive quantum evolution.

This script performs a comprehensive analysis of quantum evolution with different
scaling factors, focusing on the potential special behavior at or near the golden ratio (phi).
It generates comparative plots and metrics to visualize differences between standard
linear scaling and phi-recursive geometric scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from constants import PHI

# Import simulation components
from simulations.scripts.evolve_state import (
    run_state_evolution,
    run_phi_recursive_evolution,
    run_comparative_analysis
)
from analyses.fractal_analysis import (
    phi_sensitive_dimension,
    analyze_phi_resonance
)
from analyses.topological_invariants import (
    compute_phi_sensitive_winding,
    compute_phi_sensitive_z2,
    compute_phi_resonant_berry_phase
)

def run_phi_analysis(output_dir=None, num_qubits=1, n_steps=100):
    """
    Run comprehensive phi-resonant analysis and generate comparative plots.
    
    Parameters:
    -----------
    output_dir : Path or str, optional
        Directory to save results. If None, uses current directory.
    num_qubits : int, optional
        Number of qubits in the system (default: 1).
    n_steps : int, optional
        Number of evolution steps (default: 100).
        
    Returns:
    --------
    dict
        Dictionary containing analysis results.
    """
    if output_dir is None:
        output_dir = Path("report")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define scaling factors with dense sampling around phi
    phi = PHI
    phi_neighborhood = np.linspace(phi - 0.1, phi + 0.1, 11)
    scaling_factors = np.sort(np.concatenate([
        np.linspace(0.5, phi - 0.1, 5),
        phi_neighborhood,
        np.linspace(phi + 0.1, 3.0, 5)
    ]))
    
    print(f"Running phi-resonant analysis with {len(scaling_factors)} scaling factors...")
    print(f"Phi = {phi:.6f}")
    
    # Run comparative analysis
    results = run_comparative_analysis(
        scaling_factors=scaling_factors,
        num_qubits=num_qubits,
        state_label="phi_sensitive",
        n_steps=n_steps
    )
    
    # Extract metrics for plotting
    metrics = {
        'scaling_factors': scaling_factors,
        'state_overlaps': [],
        'dimension_differences': [],
        'phi_proximities': [],
        'standard_dimensions': [],
        'phi_dimensions': [],
        'phi_windings': [],
        'berry_phases': []
    }
    
    for factor in scaling_factors:
        # Get comparative metrics
        comp_metrics = results['comparative_metrics'][factor]
        metrics['state_overlaps'].append(comp_metrics['state_overlap'])
        metrics['dimension_differences'].append(comp_metrics['dimension_difference'])
        metrics['phi_proximities'].append(comp_metrics['phi_proximity'])
        
        # Get standard fractal dimension
        std_result = results['standard_results'][factor]
        if hasattr(std_result, 'fractal_dimensions'):
            metrics['standard_dimensions'].append(np.nanmean(std_result.fractal_dimensions))
        else:
            metrics['standard_dimensions'].append(np.nan)
        
        # Get phi-sensitive dimension
        phi_result = results['phi_recursive_results'][factor]
        if hasattr(phi_result, 'phi_dimension'):
            metrics['phi_dimensions'].append(phi_result.phi_dimension)
        else:
            metrics['phi_dimensions'].append(np.nan)
        
        # Get phi-sensitive winding
        if hasattr(phi_result, 'phi_winding'):
            metrics['phi_windings'].append(phi_result.phi_winding)
        else:
            metrics['phi_windings'].append(np.nan)
        
        # Get phi-resonant Berry phase
        if hasattr(phi_result, 'phi_berry_phase'):
            metrics['berry_phases'].append(phi_result.phi_berry_phase)
        else:
            metrics['berry_phases'].append(np.nan)
    
    # Create comparative plots
    create_comparative_plots(metrics, output_dir)
    
    # Save results to CSV
    df = pd.DataFrame({
        'Scaling Factor': metrics['scaling_factors'],
        'State Overlap': metrics['state_overlaps'],
        'Dimension Difference': metrics['dimension_differences'],
        'Phi Proximity': metrics['phi_proximities'],
        'Standard Dimension': metrics['standard_dimensions'],
        'Phi-Sensitive Dimension': metrics['phi_dimensions'],
        'Phi-Sensitive Winding': metrics['phi_windings'],
        'Berry Phase': metrics['berry_phases']
    })
    
    df.to_csv(output_dir / "phi_resonant_analysis.csv", index=False)
    print(f"Results saved to {output_dir / 'phi_resonant_analysis.csv'}")
    
    # Generate summary table
    phi_idx = np.argmin(np.abs(scaling_factors - phi))
    summary = {
        'At Phi': {
            'Scaling Factor': scaling_factors[phi_idx],
            'State Overlap': metrics['state_overlaps'][phi_idx],
            'Dimension Difference': metrics['dimension_differences'][phi_idx],
            'Standard Dimension': metrics['standard_dimensions'][phi_idx],
            'Phi-Sensitive Dimension': metrics['phi_dimensions'][phi_idx],
            'Phi-Sensitive Winding': metrics['phi_windings'][phi_idx],
            'Berry Phase': metrics['berry_phases'][phi_idx]
        },
        'Mean': {
            'State Overlap': np.nanmean(metrics['state_overlaps']),
            'Dimension Difference': np.nanmean(metrics['dimension_differences']),
            'Standard Dimension': np.nanmean(metrics['standard_dimensions']),
            'Phi-Sensitive Dimension': np.nanmean(metrics['phi_dimensions']),
            'Phi-Sensitive Winding': np.nanmean(metrics['phi_windings']),
            'Berry Phase': np.nanmean(metrics['berry_phases'])
        },
        'Max': {
            'State Overlap': np.nanmax(metrics['state_overlaps']),
            'Dimension Difference': np.nanmax(metrics['dimension_differences']),
            'Standard Dimension': np.nanmax(metrics['standard_dimensions']),
            'Phi-Sensitive Dimension': np.nanmax(metrics['phi_dimensions']),
            'Phi-Sensitive Winding': np.nanmax(metrics['phi_windings']),
            'Berry Phase': np.nanmax(metrics['berry_phases'])
        }
    }
    
    # Create summary table
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "phi_resonant_summary.csv")
    print(f"Summary saved to {output_dir / 'phi_resonant_summary.csv'}")
    
    return results

def create_comparative_plots(metrics, output_dir):
    """
    Create comparative plots of standard vs. phi-recursive evolution.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics for plotting.
    output_dir : Path
        Directory to save plots.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot fractal dimensions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['scaling_factors'], metrics['standard_dimensions'], 'o-', color='#1f77b4', label='Standard')
    ax1.plot(metrics['scaling_factors'], metrics['phi_dimensions'], 'o-', color='#ff7f0e', label='Phi-Sensitive')
    ax1.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax1.set_xlabel('Scale Factor (f_s)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot dimension difference
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['scaling_factors'], metrics['dimension_differences'], 'o-', color='#2ca02c')
    ax2.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax2.set_xlabel('Scale Factor (f_s)')
    ax2.set_ylabel('Dimension Difference')
    ax2.set_title('Fractal Dimension Difference (Phi-Sensitive - Standard)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot phi-sensitive winding
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics['scaling_factors'], metrics['phi_windings'], 'o-', color='#d62728')
    ax3.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax3.set_xlabel('Scale Factor (f_s)')
    ax3.set_ylabel('Phi-Sensitive Winding')
    ax3.set_title('Phi-Sensitive Winding Number')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot Berry phase
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics['scaling_factors'], metrics['berry_phases'], 'o-', color='#9467bd')
    ax4.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    ax4.set_xlabel('Scale Factor (f_s)')
    ax4.set_ylabel('Berry Phase')
    ax4.set_title('Phi-Resonant Berry Phase')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add overall title
    fig.suptitle('Phi-Resonant Quantum Properties', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / "phi_resonant_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir / 'phi_resonant_comparison.png'}")
    
    # Create state overlap plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['scaling_factors'], metrics['state_overlaps'], 'o-', color='#1f77b4')
    plt.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    plt.xlabel('Scale Factor (f_s)')
    plt.ylabel('State Overlap')
    plt.title('State Overlap Between Standard and Phi-Recursive Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "phi_resonant_overlap.png", dpi=300, bbox_inches='tight')
    print(f"Overlap plot saved to {output_dir / 'phi_resonant_overlap.png'}")
    
    plt.close('all')

if __name__ == "__main__":
    # Run phi-resonant analysis
    results = run_phi_analysis(output_dir="report", num_qubits=1, n_steps=100)
    
    print("\nAnalysis complete.")
    print(f"Phi = {PHI:.6f}")
    
    # Print key results at phi
    phi_idx = np.argmin(np.abs(results['scaling_factors'] - PHI))
    phi_factor = results['scaling_factors'][phi_idx]
    
    print(f"\nResults at φ ≈ {phi_factor:.6f}:")
    print(f"State overlap: {results['comparative_metrics'][phi_factor]['state_overlap']:.6f}")
    print(f"Dimension difference: {results['comparative_metrics'][phi_factor]['dimension_difference']:.6f}")
    
    # Compare with results at non-phi values
    non_phi_factors = [f for f in results['scaling_factors'] if abs(f - PHI) > 0.5]
    if non_phi_factors:
        non_phi_overlaps = [results['comparative_metrics'][f]['state_overlap'] for f in non_phi_factors]
        non_phi_dim_diffs = [results['comparative_metrics'][f]['dimension_difference'] for f in non_phi_factors]
        
        print(f"\nAverage results away from φ:")
        print(f"State overlap: {np.mean(non_phi_overlaps):.6f}")
        print(f"Dimension difference: {np.mean(non_phi_dim_diffs):.6f}")

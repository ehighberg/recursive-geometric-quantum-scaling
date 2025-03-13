#!/usr/bin/env python
# run_phi_resonant_analysis_fixed.py

"""
Fixed implementation of phi-resonant analysis to compare standard and phi-recursive
quantum evolution without artificially enhancing phi-related effects.

This script performs a comprehensive analysis of quantum evolution with different
scaling factors, focusing on potential special behavior at or near the golden ratio (phi).
It generates comparative plots and metrics with proper statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
from constants import PHI
from tqdm import tqdm
import scipy.stats as stats

# Import fixed implementations
from simulations.scripts.evolve_state_fixed import (
    run_state_evolution_fixed,
    run_phi_recursive_evolution_fixed,
    run_comparative_analysis_fixed
)
from analyses.fractal_analysis_fixed import (
    fractal_dimension,
    analyze_fractal_properties  # Renamed from analyze_phi_resonance
)

def run_phi_analysis_fixed(output_dir=None, num_qubits=1, n_steps=100):
    """
    Run comprehensive phi-resonant analysis using fixed implementations.
    
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
    
    # Define scaling factors with systematic sampling
    # Include phi, but don't sample more densely around it
    phi = PHI
    scaling_factors = np.sort(np.unique(np.concatenate([
        np.linspace(0.5, 3.0, 25),  # Uniform sampling across range
        [phi]  # Explicitly include phi
    ])))
    
    print(f"Running phi-resonant analysis with {len(scaling_factors)} scaling factors...")
    print(f"Phi = {phi:.6f}")
    
    # Run comparative analysis using fixed implementation
    results = run_comparative_analysis_fixed(
        scaling_factors=scaling_factors,
        num_qubits=num_qubits,
        state_label="phi_sensitive",
        n_steps=n_steps
    )
    
    # Extract metrics for plotting with statistical significance
    metrics = {
        'scaling_factors': scaling_factors,
        'state_overlaps': [],
        'dimension_differences': [],
        'phi_proximities': [],
        'standard_dimensions': [],
        'phi_dimensions': []
    }
    
    print("Extracting metrics for plotting...")
    for factor in tqdm(scaling_factors, desc="Processing scaling factors", unit="factor"):
        # Get comparative metrics
        comp_metrics = results['comparative_metrics'][factor]
        metrics['state_overlaps'].append(comp_metrics['state_overlap'])
        
        if 'dimension_difference' in comp_metrics:
            metrics['dimension_differences'].append(comp_metrics['dimension_difference'])
        else:
            metrics['dimension_differences'].append(np.nan)
            
        metrics['phi_proximities'].append(comp_metrics['phi_proximity'])
        
        # Get standard dimensions
        std_result = results['standard_results'][factor]
        if hasattr(std_result, 'fractal_dimensions'):
            metrics['standard_dimensions'].append(np.nanmean(std_result.fractal_dimensions))
        else:
            metrics['standard_dimensions'].append(np.nan)
        
        # Get phi-recursive dimensions
        phi_result = results['phi_recursive_results'][factor]
        if hasattr(phi_result, 'phi_dimension'):
            metrics['phi_dimensions'].append(phi_result.phi_dimension)
        else:
            metrics['phi_dimensions'].append(np.nan)
    
    # Create comparative plots with statistical significance
    print("Creating comparative plots with statistical significance...")
    create_comparative_plots_with_statistics(metrics, results['statistical_significance'], output_dir)
    
    # Save results to CSV with statistical information
    print("Saving results to CSV with statistical analysis...")
    create_comprehensive_results_csv(metrics, results['statistical_significance'], output_dir)
    
    # Create summary table with proper statistical analysis
    print("Generating summary table with statistical analysis...")
    create_summary_with_statistics(metrics, results['statistical_significance'], output_dir)
    
    return results

def create_comparative_plots_with_statistics(metrics, significance, output_dir):
    """
    Create comparative plots with error bars and statistical significance indicators.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics for plotting
    significance : dict
        Dictionary containing statistical significance analysis
    output_dir : Path
        Directory to save plots
    """
    phi = PHI
    phi_idx = np.argmin(np.abs(metrics['scaling_factors'] - phi))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot fractal dimensions with error bars and significance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['scaling_factors'], metrics['standard_dimensions'], 'o-', color='#1f77b4', label='Standard')
    ax1.plot(metrics['scaling_factors'], metrics['phi_dimensions'], 'o-', color='#ff7f0e', label='Phi-Recursive')
    ax1.axvline(x=phi, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {phi:.6f}')
    
    # Add statistical significance if available
    if 'dimension_difference' in significance and significance['dimension_difference']['significant']:
        # Highlight phi point
        phi_std = metrics['standard_dimensions'][phi_idx]
        phi_phi = metrics['phi_dimensions'][phi_idx]
        ax1.plot(phi, phi_std, 'o', color='blue', markersize=10)
        ax1.plot(phi, phi_phi, 'o', color='orange', markersize=10)
        
        # Add significance annotation
        sig_info = significance['dimension_difference']
        ax1.annotate(f"p={sig_info['p_value']:.3f}\nz={sig_info['z_score']:.2f}", 
                   xy=(phi, max(phi_std, phi_phi)),
                   xytext=(10, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->"))
    
    ax1.set_xlabel('Scale Factor (f_s)')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot dimension difference
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['scaling_factors'], metrics['dimension_differences'], 'o-', color='#2ca02c')
    ax2.axvline(x=phi, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {phi:.6f}')
    
    # Add statistical significance if available
    if 'dimension_difference' in significance:
        sig_info = significance['dimension_difference']
        if sig_info['significant']:
            # Highlight phi point
            ax2.plot(phi, metrics['dimension_differences'][phi_idx], 'o', color='red', markersize=10)
            
            # Add significance annotation
            ax2.annotate(f"p={sig_info['p_value']:.3f}\nSignificant", 
                       xy=(phi, metrics['dimension_differences'][phi_idx]),
                       xytext=(10, 10), textcoords="offset points",
                       arrowprops=dict(arrowstyle="->"))
        else:
            # Add non-significance annotation
            ax2.annotate(f"p={sig_info['p_value']:.3f}\nNot significant", 
                       xy=(phi, metrics['dimension_differences'][phi_idx]),
                       xytext=(10, 10), textcoords="offset points",
                       arrowprops=dict(arrowstyle="->"))
    
    ax2.set_xlabel('Scale Factor (f_s)')
    ax2.set_ylabel('Dimension Difference')
    ax2.set_title('Fractal Dimension Difference (Phi-Recursive - Standard)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot state overlap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics['scaling_factors'], metrics['state_overlaps'], 'o-', color='#d62728')
    ax3.axvline(x=phi, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {phi:.6f}')
    
    # Add statistical significance if available
    if 'state_overlap' in significance:
        sig_info = significance['state_overlap']
        if sig_info['significant']:
            # Highlight phi point
            ax3.plot(phi, metrics['state_overlaps'][phi_idx], 'o', color='red', markersize=10)
            
            # Add significance annotation
            ax3.annotate(f"p={sig_info['p_value']:.3f}\nSignificant", 
                       xy=(phi, metrics['state_overlaps'][phi_idx]),
                       xytext=(10, 10), textcoords="offset points",
                       arrowprops=dict(arrowstyle="->"))
    
    ax3.set_xlabel('Scale Factor (f_s)')
    ax3.set_ylabel('State Overlap')
    ax3.set_title('State Overlap Between Standard and Phi-Recursive Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot phi proximity (for reference only)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics['scaling_factors'], metrics['phi_proximities'], 'o-', color='#9467bd')
    ax4.axvline(x=phi, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {phi:.6f}')
    ax4.set_xlabel('Scale Factor (f_s)')
    ax4.set_ylabel('Phi Proximity')
    ax4.set_title('Proximity to Golden Ratio (φ)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add overall title with statistical note
    has_significance = any(sig.get('significant', False) for sig in significance.values() if isinstance(sig, dict))
    if has_significance:
        fig.suptitle('Phi-Resonant Quantum Properties (Statistical Significance Detected)', fontsize=16)
    else:
        fig.suptitle('Phi-Resonant Quantum Properties (No Statistical Significance Detected)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_dir / "phi_resonant_comparison_fixed.png", dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_dir / 'phi_resonant_comparison_fixed.png'}")
    
    # Create state overlap plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['scaling_factors'], metrics['state_overlaps'], 'o-', color='#1f77b4')
    plt.axvline(x=phi, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {phi:.6f}')
    
    # Add significance if available
    if 'state_overlap' in significance and significance['state_overlap']['significant']:
        plt.plot(phi, metrics['state_overlaps'][phi_idx], 'o', color='red', markersize=10)
        sig_info = significance['state_overlap']
        plt.annotate(f"p={sig_info['p_value']:.3f}\nz={sig_info['z_score']:.2f}", 
                   xy=(phi, metrics['state_overlaps'][phi_idx]),
                   xytext=(10, 10), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->"))
        plt.title('State Overlap Between Standard and Phi-Recursive Evolution (Statistically Significant)')
    else:
        plt.title('State Overlap Between Standard and Phi-Recursive Evolution (Not Statistically Significant)')
    
    plt.xlabel('Scale Factor (f_s)')
    plt.ylabel('State Overlap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "phi_resonant_overlap_fixed.png", dpi=300, bbox_inches='tight')
    print(f"Overlap plot saved to {output_dir / 'phi_resonant_overlap_fixed.png'}")
    
    plt.close('all')

def create_comprehensive_results_csv(metrics, significance, output_dir):
    """
    Create comprehensive results CSV with statistical analysis.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics
    significance : dict
        Dictionary containing statistical significance analysis
    output_dir : Path
        Directory to save CSV
    """
    phi = PHI
    phi_idx = np.argmin(np.abs(metrics['scaling_factors'] - phi))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Scaling Factor': metrics['scaling_factors'],
        'State Overlap': metrics['state_overlaps'],
        'Standard Dimension': metrics['standard_dimensions'],
        'Phi-Recursive Dimension': metrics['phi_dimensions'],
        'Dimension Difference': metrics['dimension_differences'],
        'Phi Proximity': metrics['phi_proximities']
    })
    
    # Add statistical significance indicators
    if 'state_overlap' in significance:
        df['State Overlap Significant'] = False
        df.loc[phi_idx, 'State Overlap Significant'] = significance['state_overlap'].get('significant', False)
        df.loc[phi_idx, 'State Overlap p-value'] = significance['state_overlap'].get('p_value', np.nan)
        df.loc[phi_idx, 'State Overlap z-score'] = significance['state_overlap'].get('z_score', np.nan)
    
    if 'dimension_difference' in significance:
        df['Dimension Difference Significant'] = False
        df.loc[phi_idx, 'Dimension Difference Significant'] = significance['dimension_difference'].get('significant', False)
        df.loc[phi_idx, 'Dimension Difference p-value'] = significance['dimension_difference'].get('p_value', np.nan)
        df.loc[phi_idx, 'Dimension Difference z-score'] = significance['dimension_difference'].get('z_score', np.nan)
    
    # Save to CSV
    output_path = output_dir / "phi_resonant_analysis_fixed.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Create metadata file with analysis details
    metadata = {
        'Implementation': 'Fixed',
        'Phi Value': float(phi),
        'Statistical Tests': 'Two-tailed t-test',
        'Significance Threshold': 0.05,
        'Scaling Factors': len(metrics['scaling_factors']),
        'Measurement Error Handling': 'Included in statistical analysis',
        'Data Collection Method': 'Unbiased, consistent algorithm across all scaling factors'
    }
    
    # Add significance summary
    for metric_name, sig_info in significance.items():
        if isinstance(sig_info, dict) and 'significant' in sig_info:
            metadata[f'{metric_name} Significant'] = sig_info['significant']
            metadata[f'{metric_name} p-value'] = sig_info['p_value']
            metadata[f'{metric_name} z-score'] = sig_info['z_score']
    
    pd.DataFrame([metadata]).to_csv(output_dir / "phi_resonant_analysis_metadata.csv", index=False)

def create_summary_with_statistics(metrics, significance, output_dir):
    """
    Create summary table with proper statistical analysis.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics
    significance : dict
        Dictionary containing statistical significance analysis
    output_dir : Path
        Directory to save summary
    """
    phi = PHI
    phi_idx = np.argmin(np.abs(metrics['scaling_factors'] - phi))
    
    # Calculate metrics for phi value
    at_phi = {
        'Scaling Factor': metrics['scaling_factors'][phi_idx],
        'State Overlap': metrics['state_overlaps'][phi_idx],
        'Standard Dimension': metrics['standard_dimensions'][phi_idx],
        'Phi-Recursive Dimension': metrics['phi_dimensions'][phi_idx],
        'Dimension Difference': metrics['dimension_differences'][phi_idx]
    }
    
    # Calculate metrics for all other values
    non_phi_indices = [i for i in range(len(metrics['scaling_factors'])) if i != phi_idx]
    mean = {
        'State Overlap': np.nanmean([metrics['state_overlaps'][i] for i in non_phi_indices]),
        'Standard Dimension': np.nanmean([metrics['standard_dimensions'][i] for i in non_phi_indices]),
        'Phi-Recursive Dimension': np.nanmean([metrics['phi_dimensions'][i] for i in non_phi_indices]),
        'Dimension Difference': np.nanmean([metrics['dimension_differences'][i] for i in non_phi_indices])
    }
    
    std = {
        'State Overlap': np.nanstd([metrics['state_overlaps'][i] for i in non_phi_indices]),
        'Standard Dimension': np.nanstd([metrics['standard_dimensions'][i] for i in non_phi_indices]),
        'Phi-Recursive Dimension': np.nanstd([metrics['phi_dimensions'][i] for i in non_phi_indices]),
        'Dimension Difference': np.nanstd([metrics['dimension_differences'][i] for i in non_phi_indices])
    }
    
    max_vals = {
        'State Overlap': np.nanmax(metrics['state_overlaps']),
        'Standard Dimension': np.nanmax(metrics['standard_dimensions']),
        'Phi-Recursive Dimension': np.nanmax(metrics['phi_dimensions']),
        'Dimension Difference': np.nanmax(metrics['dimension_differences'])
    }
    
    # Add statistical significance
    significance_info = {}
    for metric in ['State Overlap', 'Dimension Difference']:
        key = metric.lower().replace(' ', '_')
        if key in significance:
            sig_info = significance[key]
            significance_info[metric] = {
                'p_value': sig_info.get('p_value', np.nan),
                'z_score': sig_info.get('z_score', np.nan),
                'significant': sig_info.get('significant', False)
            }
    
    # Create summary DataFrame
    summary = {
        'At Phi': at_phi,
        'Mean (Non-Phi)': mean,
        'Std (Non-Phi)': std,
        'Max': max_vals,
        'Statistical Significance': significance_info
    }
    
    # Save summary to CSV
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(output_dir / "phi_resonant_summary_fixed.csv")
    print(f"Summary saved to {output_dir / 'phi_resonant_summary_fixed.csv'}")
    
    # Create formatted version for display
    html_table = summary_df.to_html()
    with open(output_dir / "phi_resonant_summary_fixed.html", "w") as f:
        f.write("<html><body>\n")
        f.write("<h1>Phi Resonant Analysis Summary (Fixed Implementation)</h1>\n")
        f.write("<p>Statistical significance threshold: p < 0.05</p>\n")
        f.write(html_table)
        f.write("</body></html>")

if __name__ == "__main__":
    # Run phi-resonant analysis
    results = run_phi_analysis_fixed(output_dir="report", num_qubits=1, n_steps=100)
    
    print("\nAnalysis complete.")
    print(f"Phi = {PHI:.6f}")
    
    # Print statistical significance results
    print("\nStatistical Significance Results:")
    for metric_name, sig_info in results['statistical_significance'].items():
        if isinstance(sig_info, dict) and 'significant' in sig_info:
            significance = "SIGNIFICANT" if sig_info['significant'] else "NOT significant"
            print(f"{metric_name}: p={sig_info['p_value']:.4f}, z={sig_info['z_score']:.4f} - {significance}")

#!/usr/bin/env python
"""
Validation script for RGQS statistical consistency.

This script analyzes the consistency between statistical validation and visualizations,
specifically focusing on the inconsistencies related to phi-resonance effects.

The primary issues addressed:
1. Fractal dimension inconsistency (shows no significance in statistical validation)
2. Visualization vs. statistical validation misalignment
3. Data validation for all metrics across different system sizes and time evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import seaborn as sns
from matplotlib.gridspec import GridSpec
from constants import PHI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import fixed implementations
from simulations.scripts.evolve_state_fixed import (
    run_state_evolution_fixed,
    run_phi_recursive_evolution_fixed,
    run_comparative_analysis_fixed
)
from analyses.fractal_analysis_fixed import (
    fractal_dimension,
    analyze_fractal_properties
)
from analyses.statistical_validation import (
    calculate_effect_size,
    run_statistical_tests,
    apply_multiple_testing_correction,
    StatisticalValidator
)

def create_output_directory(output_dir=None):
    """Create output directory for validation results."""
    if output_dir is None:
        output_dir = Path("validation")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir

def validate_fractal_dimension_analysis(scaling_factors=None):
    """
    Validate fractal dimension analysis for consistency with statistical validation.
    
    Parameters:
    -----------
    scaling_factors : List[float], optional
        Scaling factors to analyze. If None, uses a default range with phi.
        
    Returns:
    --------
    Dict
        Validation results.
    """
    logger.info("Validating fractal dimension analysis...")
    
    # Define scaling factors if not provided
    if scaling_factors is None:
        phi = PHI
        scaling_factors = np.sort(np.unique(np.concatenate([
            np.linspace(0.5, 3.0, 16),  # Regular grid
            np.linspace(phi - 0.1, phi + 0.1, 6),  # Higher density near phi
            [phi]  # Ensure phi itself is included
        ])))
    
    # Define data generation function for fractal dimension analysis
    def generate_data_for_factor(scaling_factor):
        """Generate quantum state data for a given scaling factor."""
        try:
            # Run quantum evolution with proper fixed implementation
            result = run_state_evolution_fixed(
                num_qubits=1,
                state_label="plus",
                n_steps=50,
                scaling_factor=scaling_factor
            )
            
            # Extract final state for fractal analysis
            if result and hasattr(result, 'states') and len(result.states) > 0:
                final_state = result.states[-1]
                # Convert to probability distribution
                state_data = np.abs(final_state.full().flatten())**2
                return state_data
            else:
                logger.warning(f"Invalid evolution result for factor {scaling_factor}")
                return None
        except Exception as e:
            logger.error(f"Error generating data for factor {scaling_factor}: {str(e)}")
            return None
    
    # Analyze fractal properties using fixed implementation
    logger.info("Running fractal analysis with fixed implementation...")
    fractal_results = analyze_fractal_properties(generate_data_for_factor, scaling_factors)
    
    # Extract phi index
    phi_idx = fractal_results.get('phi_index', np.argmin(np.abs(scaling_factors - PHI)))
    
    # Extract statistical analysis
    stats = fractal_results.get('statistical_analysis', {})
    
    logger.info(f"Statistical analysis results:")
    logger.info(f"  p-value: {stats.get('p_value', 'N/A')}")
    logger.info(f"  z-score: {stats.get('z_score', 'N/A')}")
    logger.info(f"  significant: {stats.get('significant', False)}")
    
    # Compute effect size between phi and other values
    # Extract dimensions
    dimensions = fractal_results.get('dimensions', [])
    
    # Calculate effect size
    phi_value = dimensions[phi_idx] if phi_idx < len(dimensions) else None
    other_values = [dim for i, dim in enumerate(dimensions) if i != phi_idx and not np.isnan(dim)]
    
    if phi_value is not None and other_values:
        effect_size = calculate_effect_size(
            np.array([phi_value]), 
            np.array(other_values)
        )
        logger.info(f"  Effect size: {effect_size:.4f}")
    else:
        effect_size = None
        logger.warning("  Could not calculate effect size due to missing values")
    
    # Create validation results
    validation_results = {
        'scaling_factors': scaling_factors,
        'fractal_dimensions': dimensions,
        'phi_index': phi_idx,
        'phi_value': PHI,
        'phi_dimension': phi_value,
        'statistical_significance': stats.get('significant', False),
        'p_value': stats.get('p_value', None),
        'z_score': stats.get('z_score', None),
        'effect_size': effect_size
    }
    
    return validation_results

def validate_phase_diagram_consistency(fractal_results, output_dir):
    """
    Validate consistency between phase diagram and statistical validation.
    
    Parameters:
    -----------
    fractal_results : Dict
        Results from fractal dimension analysis.
    output_dir : Path
        Directory to save validation results.
        
    Returns:
    --------
    Dict
        Validation results.
    """
    logger.info("Validating phase diagram consistency...")
    
    # Load phase diagram data
    phase_diagram_path = Path("paper_graphs/phase_diagram_summary.csv")
    if phase_diagram_path.exists():
        phase_df = pd.read_csv(phase_diagram_path)
        logger.info(f"Loaded phase diagram from {phase_diagram_path}")
    else:
        # Recreate phase diagram data based on paper_graphs/generate_paper_graphs.py
        logger.warning(f"Phase diagram file not found at {phase_diagram_path}, recreating...")
        phase_diagram = [
            {'f_s Range': 'f_s < 0.8', 'Phase Type': 'Trivial', 'Topological Invariant': '0', 'Fractal Dimension': 'Low (~0.8-1.0)', 'Gap Size': 'Large'},
            {'f_s Range': '0.8 < f_s < 1.4', 'Phase Type': 'Weakly Topological', 'Topological Invariant': '±1', 'Fractal Dimension': 'Medium (~1.0-1.2)', 'Gap Size': 'Medium'},
            {'f_s Range': f'f_s ≈ φ ({PHI:.6f})', 'Phase Type': 'Strongly Topological', 'Topological Invariant': '±1', 'Fractal Dimension': 'High (~1.2-1.5)', 'Gap Size': 'Small'},
            {'f_s Range': '1.8 < f_s < 2.4', 'Phase Type': 'Weakly Topological', 'Topological Invariant': '±1', 'Fractal Dimension': 'Medium (~1.0-1.2)', 'Gap Size': 'Medium'},
            {'f_s Range': 'f_s > 2.4', 'Phase Type': 'Trivial', 'Topological Invariant': '0', 'Fractal Dimension': 'Low (~0.8-1.0)', 'Gap Size': 'Large'},
        ]
        phase_df = pd.DataFrame(phase_diagram)
    
    # Extract phi row
    phi_row = phase_df[phase_df['f_s Range'].str.contains('φ', regex=False)]
    
    # Extract claimed fractal dimension range
    claimed_fractal_dim = phi_row['Fractal Dimension'].iloc[0] if len(phi_row) > 0 else "Not specified"
    logger.info(f"Phase diagram claims fractal dimension at phi: {claimed_fractal_dim}")
    
    # Extract actual fractal dimension and significance
    actual_fractal_dim = fractal_results.get('phi_dimension', None)
    significant = fractal_results.get('statistical_significance', False)
    p_value = fractal_results.get('p_value', None)
    
    logger.info(f"Actual fractal dimension at phi: {actual_fractal_dim:.4f}")
    logger.info(f"Statistically significant: {significant} (p={p_value:.4f})")
    
    # Determine if there's a consistency issue
    consistency_issue = False
    issue_description = ""
    
    # Parse claimed fractal dimension range
    import re
    claimed_range = re.search(r'~\(([\d\.]+)-([\d\.]+)\)', claimed_fractal_dim)
    if claimed_range:
        lower_bound = float(claimed_range.group(1))
        upper_bound = float(claimed_range.group(2))
        
        # Check if actual value is in claimed range
        if actual_fractal_dim is not None:
            if not (lower_bound <= actual_fractal_dim <= upper_bound):
                consistency_issue = True
                issue_description += f"Actual fractal dimension {actual_fractal_dim:.4f} is outside claimed range {lower_bound}-{upper_bound}. "
    
    # Check if significance is inconsistent with claims
    if "High" in claimed_fractal_dim and not significant:
        consistency_issue = True
        issue_description += f"Phase diagram claims 'High' fractal dimension but effect is not statistically significant (p={p_value:.4f}). "
    
    # Check if effect size supports claims
    effect_size = fractal_results.get('effect_size', None)
    if effect_size is not None:
        if abs(effect_size) < 0.2 and "High" in claimed_fractal_dim:
            consistency_issue = True
            issue_description += f"Negligible effect size ({effect_size:.4f}) doesn't support 'High' fractal dimension claim. "
    
    # Create visualization of the issue
    plt.figure(figsize=(10, 6))
    
    # Plot fractal dimensions
    scaling_factors = fractal_results.get('scaling_factors', [])
    dimensions = fractal_results.get('fractal_dimensions', [])
    
    if scaling_factors and dimensions:
        plt.plot(scaling_factors, dimensions, 'o-', label='Fractal Dimension')
        
        # Highlight phi
        phi_idx = fractal_results.get('phi_index', np.argmin(np.abs(scaling_factors - PHI)))
        plt.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
        
        if phi_idx < len(dimensions):
            plt.plot(PHI, dimensions[phi_idx], 'ro', markersize=10)
        
        # Add claimed range
        if claimed_range:
            plt.axhspan(lower_bound, upper_bound, color='yellow', alpha=0.3, 
                       label=f'Claimed range at φ ({lower_bound}-{upper_bound})')
        
        plt.xlabel('Scaling Factor (f_s)')
        plt.ylabel('Fractal Dimension')
        plt.title('Fractal Dimension vs. Scaling Factor')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistical annotation
        if p_value is not None:
            significance_text = f"p-value: {p_value:.4f}\n"
            if effect_size is not None:
                significance_text += f"Effect size: {effect_size:.4f}\n"
            significance_text += f"Statistically significant: {significant}"
            
            plt.annotate(significance_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / "fractal_dimension_validation.png", dpi=300)
        plt.close()
        
        logger.info(f"Validation plot saved to {output_dir / 'fractal_dimension_validation.png'}")
    
    # Create validation results
    validation_results = {
        'claimed_fractal_dim': claimed_fractal_dim,
        'actual_fractal_dim': actual_fractal_dim,
        'significant': significant,
        'p_value': p_value,
        'effect_size': effect_size,
        'consistency_issue': consistency_issue,
        'issue_description': issue_description
    }
    
    # Generate detailed report for the user
    if consistency_issue:
        logger.warning(f"Consistency issue detected: {issue_description}")
    else:
        logger.info("No consistency issues detected in phase diagram.")
    
    return validation_results

def validate_statistical_significance_metrics(output_dir):
    """
    Validate the statistical significance of all metrics.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save validation results.
        
    Returns:
    --------
    Dict
        Validation results.
    """
    logger.info("Validating statistical significance of all metrics...")
    
    # Load statistical validation data
    stat_validation_path = Path("paper_graphs/statistical_validation.csv")
    if not stat_validation_path.exists():
        logger.error(f"Statistical validation file not found at {stat_validation_path}")
        return None
    
    # Load data
    stat_df = pd.read_csv(stat_validation_path)
    logger.info(f"Loaded statistical validation from {stat_validation_path}")
    
    # Extract metrics and their significance
    metrics = stat_df['Metric'].tolist()
    p_values = stat_df['p-value'].tolist()
    adjusted_p = stat_df['Adjusted p-value'].tolist()
    effect_sizes = stat_df['Effect Size'].tolist()
    effect_categories = stat_df['Effect Category'].tolist()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Sort metrics by p-value
    sorted_indices = np.argsort(p_values)
    sorted_metrics = [metrics[i] for i in sorted_indices]
    sorted_p = [p_values[i] for i in sorted_indices]
    sorted_adj_p = [adjusted_p[i] for i in sorted_indices]
    
    # Create bar chart of p-values
    x = np.arange(len(sorted_metrics))
    width = 0.35
    
    # Plot raw p-values
    plt.bar(x - width/2, sorted_p, width, label='Raw p-value', color='skyblue')
    
    # Plot adjusted p-values
    plt.bar(x + width/2, sorted_adj_p, width, label='Adjusted p-value', color='salmon')
    
    # Add significance threshold line
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (α=0.05)')
    
    # Add labels showing effect sizes
    for i, metric in enumerate(sorted_metrics):
        idx = metrics.index(metric)
        effect_size = effect_sizes[idx]
        plt.annotate(f"{effect_size:.2f}",
                    xy=(i, sorted_p[i]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Configure plot
    plt.yscale('log')  # Use log scale for better visualization
    plt.xlabel('Metrics')
    plt.ylabel('p-value (log scale)')
    plt.title('Statistical Significance of Metrics')
    plt.xticks(x, sorted_metrics, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "metrics_significance_validation.png", dpi=300)
    plt.close()
    
    logger.info(f"Metrics significance validation plot saved to {output_dir / 'metrics_significance_validation.png'}")
    
    # Check for inconsistency between statistical significance and effect sizes
    inconsistencies = []
    
    for i, metric in enumerate(metrics):
        p = p_values[i]
        effect = effect_sizes[i]
        category = effect_categories[i]
        
        # Check for metrics with significant p-value but negligible effect size
        if p < 0.05 and category == 'negligible':
            inconsistencies.append(f"{metric}: significant p-value ({p:.4f}) but negligible effect size ({effect:.4f})")
        
        # Check for metrics with large effect size but non-significant p-value
        if p >= 0.05 and category == 'large':
            inconsistencies.append(f"{metric}: large effect size ({effect:.4f}) but non-significant p-value ({p:.4f})")
    
    # Create validation results
    validation_results = {
        'metrics': metrics,
        'p_values': p_values,
        'adjusted_p_values': adjusted_p,
        'effect_sizes': effect_sizes,
        'effect_categories': effect_categories,
        'inconsistencies': inconsistencies
    }
    
    # Report inconsistencies
    if inconsistencies:
        logger.warning(f"Inconsistencies detected between statistical significance and effect sizes:")
        for inconsistency in inconsistencies:
            logger.warning(f"  - {inconsistency}")
    else:
        logger.info("No inconsistencies detected between statistical significance and effect sizes.")
    
    return validation_results

def validate_system_size_sensitivity(output_dir, system_sizes=None):
    """
    Validate sensitivity of phi-resonance effects to system size.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save validation results.
    system_sizes : List[int], optional
        System sizes to analyze. If None, uses [1, 2, 3, 4].
        
    Returns:
    --------
    Dict
        Validation results.
    """
    logger.info("Validating system size sensitivity...")
    
    # Define system sizes if not provided
    if system_sizes is None:
        system_sizes = [1, 2, 3, 4]
    
    # Define scaling factors
    phi = PHI
    scaling_factors = [0.8, 1.0, 1.4, phi, 2.0, 2.4]
    
    # Initialize results
    results = {
        'system_sizes': system_sizes,
        'scaling_factors': scaling_factors,
        'fractal_dimensions': {},
        'statistical_tests': {}
    }
    
    # For each system size, compute fractal dimensions
    for size in system_sizes:
        logger.info(f"Analyzing system size: {size}")
        
        # Initialize results for this size
        fractal_dims = {}
        
        # For each scaling factor, run simulation
        for factor in scaling_factors:
            try:
                # Run evolution
                result = run_state_evolution_fixed(
                    num_qubits=size,
                    state_label="plus",
                    n_steps=30,  # Reduced for efficiency
                    scaling_factor=factor
                )
                
                # Calculate fractal dimension
                final_state = result.states[-1]
                state_data = np.abs(final_state.full().flatten())**2
                fd = fractal_dimension(state_data)
                
                # Store result
                fractal_dims[factor] = fd
                logger.info(f"  Scaling factor {factor:.6f}: dimension = {fd:.4f}")
                
            except Exception as e:
                logger.error(f"Error for system size {size}, factor {factor}: {str(e)}")
                fractal_dims[factor] = np.nan
        
        # Store dimensions for this size
        results['fractal_dimensions'][size] = fractal_dims
        
        # Perform statistical test between phi and other factors
        phi_data = [fractal_dims[phi]] if not np.isnan(fractal_dims[phi]) else []
        other_data = [fractal_dims[f] for f in scaling_factors if f != phi and not np.isnan(fractal_dims[f])]
        
        if phi_data and other_data:
            # Run statistical test
            test_result = run_statistical_tests(np.array(phi_data), np.array(other_data))
            results['statistical_tests'][size] = test_result
            
            logger.info(f"  Statistical test for size {size}:")
            logger.info(f"    p-value: {test_result['t_test']['p_value']:.4f}")
            logger.info(f"    effect size: {test_result['effect_size']:.4f}")
            logger.info(f"    significant: {test_result['significant']}")
        else:
            logger.warning(f"  Insufficient data for statistical test for size {size}")
            results['statistical_tests'][size] = None
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # For each system size, plot fractal dimensions
    for size in system_sizes:
        # Extract dimensions for this size
        dims = [results['fractal_dimensions'][size].get(f, np.nan) for f in scaling_factors]
        
        # Plot dimensions
        plt.plot(scaling_factors, dims, 'o-', label=f'{size} qubit{"s" if size > 1 else ""}')
    
    # Add phi line
    plt.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.6f}')
    
    # Configure plot
    plt.xlabel('Scaling Factor (f_s)')
    plt.ylabel('Fractal Dimension')
    plt.title('Fractal Dimension vs. System Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "system_size_sensitivity.png", dpi=300)
    plt.close()
    
    logger.info(f"System size sensitivity plot saved to {output_dir / 'system_size_sensitivity.png'}")
    
    # Create summary of p-values and effect sizes
    p_values = {}
    effect_sizes = {}
    significance = {}
    
    for size, test_result in results['statistical_tests'].items():
        if test_result:
            p_values[size] = test_result['t_test']['p_value']
            effect_sizes[size] = test_result['effect_size']
            significance[size] = test_result['significant']
    
    # Check consistency of significance across system sizes
    significant_sizes = [size for size, sig in significance.items() if sig]
    non_significant_sizes = [size for size, sig in significance.items() if not sig]
    
    consistency_issue = False
    issue_description = ""
    
    if significant_sizes and non_significant_sizes:
        consistency_issue = True
        issue_description = f"Inconsistent significance: significant for sizes {significant_sizes}, not significant for sizes {non_significant_sizes}"
    
    # Create validation summary
    validation_summary = {
        'p_values': p_values,
        'effect_sizes': effect_sizes,
        'significance': significance,
        'consistency_issue': consistency_issue,
        'issue_description': issue_description
    }
    
    # Report consistency issue
    if consistency_issue:
        logger.warning(f"System size consistency issue: {issue_description}")
    else:
        logger.info("No system size consistency issues detected.")
    
    # Add validation summary to results
    results['validation_summary'] = validation_summary
    
    return results

def validate_temporal_consistency(output_dir):
    """
    Validate temporal consistency of phi-resonance effects during evolution.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save validation results.
        
    Returns:
    --------
    Dict
        Validation results.
    """
    logger.info("Validating temporal consistency...")
    
    # Define scaling factors
    phi = PHI
    scaling_factors = [0.8, 1.0, 1.4, phi, 2.0, 2.4]
    
    # Define time steps
    n_steps = 40
    
    # Run simulations
    results = {
        'scaling_factors': scaling_factors,
        'time_steps': np.arange(n_steps),
        'states': {},
        'fractal_dimensions': {},
        'statistical_tests': {}
    }
    
    # For each scaling factor, run simulation
    for factor in scaling_factors:
        logger.info(f"Running simulation for scaling factor {factor:.6f}...")
        
        try:
            # Run evolution
            result = run_state_evolution_fixed(
                num_qubits=2,  # Using 2 qubits for more interesting dynamics
                state_label="bell",
                n_steps=n_steps,
                scaling_factor=factor
            )
            
            # Store states
            results['states'][factor] = result.states
            
            # Calculate fractal dimensions for each time step
            fractal_dims = []
            
            for state in result.states:
                state_data = np.abs(state.full().flatten())**2
                fd = fractal_dimension(state_data)
                fractal_dims.append(fd)
            
            results['fractal_dimensions'][factor] = fractal_dims
            
        except Exception as e:
            logger.error(f"Error for factor {factor}: {str(e)}")
            results['states'][factor] = None
            results['fractal_dimensions'][factor] = [np.nan] * n_steps
    
    # For each time step, perform statistical test
    for t in range(n_steps):
        # Extract fractal dimensions at this time step
        phi_dim = results['fractal_dimensions'][phi][t] if phi in results['fractal_dimensions'] else np.nan
        other_dims = [results['fractal_dimensions'][f][t] for f in scaling_factors if f != phi and f in results['fractal_dimensions']]
        
        # Perform statistical test
        if not np.isnan(phi_dim) and other_dims and not all(np.isnan(dim) for dim in other_dims):
            # Run statistical test
            test_result = run_statistical_tests(np.array([phi_dim]), np.array([d for d in other_dims if not np.isnan(d)]))
            results['statistical_tests'][t] = test_result
        else:
            results['statistical_tests'][t] = None
    
    # Create visualization of fractal dimensions over time
    plt.figure(figsize=(12, 6))
    
    # For each scaling factor, plot fractal dimensions over time
    for factor in scaling_factors:
        if factor in results['fractal_dimensions']:
            # Plot dimensions
            plt.plot(np.arange(n_steps), results['fractal_dimensions'][factor], 
                    label=f'f_s = {factor:.4f}' + (' (φ)' if factor == phi else ''))
    
    # Configure plot
    plt.xlabel('Time Step')
    plt.ylabel('Fractal Dimension')
    plt.title('Fractal Dimension Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_fractal_evolution.png", dpi=300)
    plt.close()
    
    logger.info(f"Temporal fractal evolution plot saved to {output_dir / 'temporal_fractal_evolution.png'}")
    
    # Create plot of statistical significance over time
    plt.figure(figsize=(12, 6))
    
    # Extract p-values and significance
    p_values = []
    significance = []
    
    for t in range(n_steps):
        if t in results['statistical_tests'] and results['statistical_tests'][t]:
            p_values.append(results['statistical_tests'][t]['t_test']['p_value'])
            significance.append(results['statistical_tests'][t]['significant'])
        else:
            p_values.append(np.nan)
            significance.append(False)
    
    # Plot p-values
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_steps), p_values, 'o-')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (α=0.05)')
    plt.xlabel('Time Step')
    plt.ylabel('p-value')
    plt.title('Statistical Significance Over Time')
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot significance
    plt.subplot(2, 1, 2)
    plt.step(np.arange(n_steps), significance, where='mid')
    plt.xlabel('Time Step')
    plt.ylabel('Significant')
    plt.yticks([0, 1], ['False', 'True'])
    plt.title('Significance of Phi Effect Over Time')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_significance.png", dpi=300)
    plt.close()
    
    logger.info(f"Temporal significance plot saved to {output_dir / 'temporal_significance.png'}")
    
    # Calculate temporal consistency
    valid_tests = [t for t in range(n_steps) if t in results['statistical_tests'] and results['statistical_tests'][t]]
    if valid_tests:
        significant_steps = sum(results['statistical_tests'][t]['significant'] for t in valid_tests)
        total_steps = len(valid_tests)
        consistency_ratio = significant_steps / total_steps
        
        logger.info(f"Temporal consistency ratio: {consistency_ratio:.4f} ({significant_steps}/{total_steps} steps significant)")
        
        # Determine if there's a consistency issue
        consistency_issue = False
        issue_description = ""
        
        if 0.1 < consistency_ratio < 0.9:
            consistency_issue = True
            issue_description = f"Inconsistent significance over time: only {consistency_ratio:.2f} of time steps show significant effect"
        
        # Create validation results
        validation_results = {
            'consistency_ratio': consistency_ratio,
            'significant_steps': significant_steps,
            'total_steps': total_steps,
            'consistency_issue': consistency_issue,
            'issue_description': issue_description
        }
        
        # Report consistency issue
        if consistency_issue:
            logger.warning(f"Temporal consistency issue: {issue_description}")
        else:
            logger.info("No temporal consistency issues detected.")
        
        return validation_results
    else:
        logger.warning("Insufficient data for temporal consistency analysis")
        return {
            'consistency_ratio': 0.0,
            'significant_steps': 0,
            'total_steps': 0,
            'consistency_issue': False,
            'issue_description': "Insufficient data for analysis"
        }


def main():
    """
    Main function to run all validations.
    """
    # Create output directory
    output_dir = create_output_directory()
    
    # Run fractal dimension validation
    logger.info("Starting fractal dimension validation...")
    fractal_results = validate_fractal_dimension_analysis()
    
    # Validate phase diagram consistency
    logger.info("Validating phase diagram consistency...")
    phase_results = validate_phase_diagram_consistency(fractal_results, output_dir)
    
    # Validate statistical significance metrics
    logger.info("Validating statistical significance of all metrics...")
    metrics_results = validate_statistical_significance_metrics(output_dir)
    
    # Validate system size sensitivity
    logger.info("Validating system size sensitivity...")
    system_size_results = validate_system_size_sensitivity(output_dir)
    
    # Validate temporal consistency
    logger.info("Validating temporal consistency...")
    temporal_results = validate_temporal_consistency(output_dir)
    
    # Generate summary report
    logger.info("Generating validation summary...")
    validation_summary = {
        'fractal_dimension': fractal_results,
        'phase_diagram': phase_results,
        'statistical_metrics': metrics_results,
        'system_size': system_size_results,
        'temporal_consistency': temporal_results
    }
    
    # Identify key issues
    issues = []
    
    # Check for fractal dimension issue
    if not fractal_results.get('statistical_significance', False) and 'High' in phase_results.get('claimed_fractal_dim', ''):
        issues.append("Fractal dimension is claimed to be 'High' at phi but is not statistically significant")
    
    # Check for system size consistency issue
    if system_size_results.get('validation_summary', {}).get('consistency_issue', False):
        issues.append(system_size_results['validation_summary']['issue_description'])
    
    # Check for temporal consistency issue
    if temporal_results.get('consistency_issue', False):
        issues.append(temporal_results['issue_description'])
    
    # Print summary
    if issues:
        logger.warning("Validation found the following issues:")
        for i, issue in enumerate(issues):
            logger.warning(f"{i+1}. {issue}")
    else:
        logger.info("No significant issues found in validation")
    
    logger.info("Validation complete. Results saved to validation directory.")
    
    return validation_summary


if __name__ == "__main__":
    main()

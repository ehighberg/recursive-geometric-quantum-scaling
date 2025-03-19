#!/usr/bin/env python
"""
Update analysis scripts to use the scientific validator instead of the statistical validator.

This script demonstrates how to use the ScientificValidator for more rigorous
and deterministic statistical analysis of phi-related phenomena.
"""
from analyses.scientific_validation import ScientificValidator
import numpy as np
import matplotlib.pyplot as plt
from constants import PHI
from pathlib import Path
import pandas as pd

def generate_sample_data():
    """Generate sample data for demonstration."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create test data with a modestly significant effect size
    phi_data = np.random.normal(loc=1.0, scale=0.2, size=30)  # Higher mean
    control_data1 = np.random.normal(loc=0.8, scale=0.2, size=30)  # Medium effect size
    control_data2 = np.random.normal(loc=0.7, scale=0.2, size=30)  # Large effect size
    
    # Create metrics data
    metrics_data = {
        'Fractal Dimension': {
            PHI: phi_data,
            1.0: control_data1,
            2.0: control_data2
        },
        'Topological Protection': {
            PHI: phi_data * 1.2,  # Stronger effect
            1.0: control_data1 * 1.1,
            2.0: control_data2
        },
        'Quantum Coherence': {
            PHI: phi_data * 0.9,  # Weaker effect
            1.0: control_data1 * 0.85,
            2.0: control_data2 * 0.8
        },
        'Minimal Effect': {  # Near-zero effect size
            PHI: control_data1 * 1.01,
            1.0: control_data1,
            2.0: control_data1 * 0.99
        }
    }
    
    return metrics_data

def run_scientific_validation():
    """Run scientific validation on the sample data."""
    # Generate sample data
    metrics_data = generate_sample_data()
    
    # Initialize validator
    validator = ScientificValidator(alpha=0.05)
    
    # Run validation with different correction methods
    bonferroni_results = validator.validate_multiple_metrics(
        metrics_data, correction_method='bonferroni'
    )
    
    holm_results = validator.validate_multiple_metrics(
        metrics_data, correction_method='holm'
    )
    
    bh_results = validator.validate_multiple_metrics(
        metrics_data, correction_method='benjamini-hochberg'
    )
    
    # Print results
    print("=== Scientific Validation Results ===")
    print("\nIndividual metrics (Bonferroni correction):")
    for metric, result in bonferroni_results['individual_results'].items():
        ci = result['confidence_interval']
        print(f"{metric}:")
        print(f"  p-value: {result['p_value']:.6f}, adjusted: {result['adjusted_p_value']:.6f}")
        print(f"  Effect size: {result['effect_size']:.3f} ({result['effect_size_category']})")
        print(f"  95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
        print(f"  Significant after correction: {result['significant_after_correction']}")
    
    print("\nSignificant metrics by correction method:")
    print(f"  Bonferroni: {bonferroni_results['combined_results']['significant_metrics']}")
    print(f"  Holm: {holm_results['combined_results']['significant_metrics']}")
    print(f"  Benjamini-Hochberg: {bh_results['combined_results']['significant_metrics']}")
    
    # Create results visualization
    visualize_results(bonferroni_results, metrics_data)
    
    # Save results to CSV
    save_results_to_csv(bonferroni_results, "validation_results/bonferroni_results.csv")
    save_results_to_csv(holm_results, "validation_results/holm_results.csv")
    save_results_to_csv(bh_results, "validation_results/benjamini_hochberg_results.csv")
    
    return bonferroni_results

def save_results_to_csv(results, output_path):
    """Save validation results to CSV."""
    # Create output directory if needed
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Extract individual results
    data = []
    for metric, result in results['individual_results'].items():
        ci = result['confidence_interval']
        data.append({
            'Metric': metric,
            'p_value': result['p_value'],
            'adjusted_p_value': result['adjusted_p_value'],
            'effect_size': result['effect_size'],
            'effect_size_category': result['effect_size_category'],
            'CI_lower': ci[0],
            'CI_upper': ci[1],
            'significant': result['significant_after_correction']
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def visualize_results(results, metrics_data):
    """Visualize validation results."""
    # Create output directory
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create bar plot of effect sizes with error bars
    plt.figure(figsize=(10, 6))
    
    metrics = list(results['individual_results'].keys())
    effect_sizes = [results['individual_results'][metric]['effect_size'] for metric in metrics]
    ci_errors = [
        [results['individual_results'][metric]['effect_size'] - results['individual_results'][metric]['confidence_interval'][0] 
         for metric in metrics],
        [results['individual_results'][metric]['confidence_interval'][1] - results['individual_results'][metric]['effect_size'] 
         for metric in metrics]
    ]
    
    # Create colors based on significance
    colors = ['green' if results['individual_results'][metric]['significant_after_correction'] else 'red' 
              for metric in metrics]
    
    # Plot with error bars
    plt.bar(metrics, effect_sizes, yerr=ci_errors, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add effect size thresholds
    plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
    plt.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium Effect')
    plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large Effect')
    
    plt.xlabel('Metrics')
    plt.ylabel('Effect Size (Cohen\'s d)')
    plt.title('Effect Sizes with 95% Confidence Intervals (After Bonferroni Correction)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "effect_sizes.png", dpi=300)
    
    # Create boxplot for each metric to compare phi vs. control
    plt.figure(figsize=(15, 8))
    
    # Create subplots for each metric
    nrows = 2
    ncols = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data
        phi_data = metrics_data[metric][PHI]
        control_data = []
        for factor, data in metrics_data[metric].items():
            if factor != PHI:
                control_data.extend(data)
        
        # Create boxplot
        boxplot_data = [phi_data, control_data]
        ax.boxplot(boxplot_data)
        
        # Add scatter points for individual values
        x_jitter = np.random.normal(1, 0.04, size=len(phi_data))
        ax.scatter(x_jitter, phi_data, alpha=0.3, color='green')
        
        x_jitter = np.random.normal(2, 0.04, size=len(control_data))
        ax.scatter(x_jitter, control_data, alpha=0.3, color='red')
        
        # Set labels
        ax.set_title(metric)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Phi', 'Control'])
        
        # Highlight significance
        if results['individual_results'][metric]['significant_after_correction']:
            ax.text(0.5, 0.05, 'Significant', transform=ax.transAxes,
                   fontsize=11, ha='center', color='green', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.05, 'Not Significant', transform=ax.transAxes,
                   fontsize=11, ha='center', color='red', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Add p-value
        p_value = results['individual_results'][metric]['adjusted_p_value']
        ax.text(0.5, 0.95, f'p = {p_value:.4f}', transform=ax.transAxes,
               fontsize=10, ha='center', 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparisons.png", dpi=300)
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    results = run_scientific_validation()

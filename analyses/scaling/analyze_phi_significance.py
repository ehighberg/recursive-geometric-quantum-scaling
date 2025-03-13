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

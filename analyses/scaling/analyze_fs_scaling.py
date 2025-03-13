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

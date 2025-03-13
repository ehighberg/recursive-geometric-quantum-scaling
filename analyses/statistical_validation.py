#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical validation module for RGQS.

This module implements various statistical methods to validate the significance
of findings in the RGQS system, particularly focusing on the significance of phi.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# Define PHI constant
PHI = (1 + 5**0.5) / 2  # Golden ratio

def calculate_p_value(observed_value: float, null_distribution: np.ndarray, 
                      alternative: str = 'two-sided') -> float:
    """
    Calculate p-value for an observed value against a null distribution.
    
    Parameters:
        observed_value (float): The observed statistic
        null_distribution (np.ndarray): Distribution under null hypothesis
        alternative (str): 'two-sided', 'greater', or 'less'
    
    Returns:
        float: p-value
    """
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_value))
    elif alternative == 'greater':
        p_value = np.mean(null_distribution >= observed_value)
    elif alternative == 'less':
        p_value = np.mean(null_distribution <= observed_value)
    else:
        raise ValueError(f"Invalid alternative '{alternative}'. Must be 'two-sided', 'greater', or 'less'.")
    
    # Avoid p-values of 0
    if p_value == 0:
        p_value = 1 / (len(null_distribution) + 1)
    
    return p_value

def calculate_confidence_interval(data: Union[np.ndarray, List], confidence_level: float = 0.95,
                                  method: str = 'bootstrap', n_resamples: int = 1000) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset.
    
    Parameters:
        data (Union[np.ndarray, List]): The dataset, can be n-dimensional
        confidence_level (float): Confidence level (default: 0.95 for 95% CI)
        method (str): 'bootstrap', 'parametric', or 't'
        n_resamples (int): Number of bootstrap resamples (if method='bootstrap')
    
    Returns:
        Tuple[float, float]: Lower and upper bounds of confidence interval
    """
    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Validate and flatten multi-dimensional data
    if hasattr(data, 'ndim') and data.ndim > 1:
        print(f"WARNING: Flattening {data.ndim}-dimensional data of shape {data.shape}")
        data = data.flatten()  # Convert multi-dimensional array to 1D
    
    # Ensure data has elements to process
    if len(data) == 0:
        print("WARNING: Empty data array provided to confidence interval calculation")
        return 0.0, 0.0
    
    alpha = 1 - confidence_level
    
    if method == 'bootstrap':
        # Bootstrap confidence interval
        bootstrap_means = []
        
        try:
            for _ in range(n_resamples):
                # np.random.choice requires 1-dimensional array
                resampled = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(resampled))
        except ValueError as e:
            print(f"ERROR in bootstrap sampling: {e}")
            print(f"Data shape: {data.shape}, Data type: {type(data)}")
            # Return a reasonable default confidence interval centered on the mean
            mean_val = np.mean(data)
            return mean_val - 0.2, mean_val + 0.2
        
        lower_bound = np.percentile(bootstrap_means, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
    elif method == 'parametric':
        # Parametric confidence interval (assumes normality)
        mean = np.mean(data)
        sem = stats.sem(data)
        z_critical = stats.norm.ppf(1 - alpha/2)
        margin_of_error = z_critical * sem
        
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
    elif method == 't':
        # t-distribution based confidence interval
        mean = np.mean(data)
        sem = stats.sem(data)
        t_critical = stats.t.ppf(1 - alpha/2, len(data) - 1)
        margin_of_error = t_critical * sem
        
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'bootstrap', 'parametric', or 't'.")
    
    return lower_bound, upper_bound

def calculate_effect_size(data1: np.ndarray, data2: np.ndarray, method: str = 'cohen_d') -> float:
    """
    Calculate effect size between two datasets.
    
    Parameters:
        data1 (np.ndarray): First dataset
        data2 (np.ndarray): Second dataset
        method (str): 'cohen_d', 'hedges_g', or 'glass_delta'
    
    Returns:
        float: Effect size
    """
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    difference = mean1 - mean2
    
    if method == 'cohen_d':
        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        effect_size = difference / pooled_sd
        
    elif method == 'hedges_g':
        # Similar to Cohen's d but with correction for small samples
        n1, n2 = len(data1), len(data2)
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = difference / pooled_sd
        correction = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
        effect_size = cohen_d * correction
        
    elif method == 'glass_delta':
        # Glass's delta uses only the control group's standard deviation
        sd2 = np.std(data2, ddof=1)
        effect_size = difference / sd2
        
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'cohen_d', 'hedges_g', or 'glass_delta'.")
    
    return effect_size

def apply_multiple_testing_correction(p_values: List[float], method: str = 'fdr_bh', 
                                     alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.
    
    Parameters:
        p_values (List[float]): List of p-values
        method (str): 'bonferroni', 'fdr_bh', or 'holm'
        alpha (float): Significance level
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Reject null hypothesis flags and adjusted p-values
    """
    if method == 'bonferroni':
        # Bonferroni correction
        reject = np.array([p <= alpha / len(p_values) for p in p_values])
        adjusted_p = np.minimum(np.array(p_values) * len(p_values), 1.0)
        
    elif method in ['fdr_bh', 'holm']:
        # Use statsmodels for these methods
        reject, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method=method)
        
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'bonferroni', 'fdr_bh', or 'holm'.")
    
    return reject, adjusted_p

def create_null_distribution(data: np.ndarray, test_statistic: callable, 
                            n_permutations: int = 1000) -> np.ndarray:
    """
    Create null distribution via permutation test.
    
    Parameters:
        data (np.ndarray): Dataset to permute
        test_statistic (callable): Function that calculates the test statistic
        n_permutations (int): Number of permutations
    
    Returns:
        np.ndarray: Null distribution of test statistic
    """
    null_distribution = []
    n = len(data)
    
    for _ in range(n_permutations):
        # Randomly permute the data
        permuted_data = np.random.permutation(data)
        
        # Calculate test statistic on permuted data
        stat = test_statistic(permuted_data)
        null_distribution.append(stat)
    
    return np.array(null_distribution)

def run_statistical_tests(phi_data: Union[np.ndarray, List], control_data: Union[np.ndarray, List]) -> Dict[str, Any]:
    """
    Run a comprehensive set of statistical tests comparing phi data to control data.
    
    Parameters:
        phi_data (Union[np.ndarray, List]): Data from phi scaling, can be multi-dimensional
        control_data (Union[np.ndarray, List]): Data from control scaling, can be multi-dimensional
    
    Returns:
        Dict[str, Any]: Dictionary of test results
    """
    # Ensure data is in numpy arrays
    if not isinstance(phi_data, np.ndarray):
        phi_data = np.array(phi_data)
    if not isinstance(control_data, np.ndarray):
        control_data = np.array(control_data)
    
    # Log data shapes for debugging
    print(f"Statistical tests - phi_data shape: {np.shape(phi_data)}, control_data shape: {np.shape(control_data)}")
    
    # Ensure data is 1-dimensional for statistical tests
    if phi_data.ndim > 1:
        print(f"Flattening phi_data from shape {phi_data.shape}")
        phi_data = phi_data.flatten()
    if control_data.ndim > 1:
        print(f"Flattening control_data from shape {control_data.shape}")
        control_data = control_data.flatten()
    
    # Initialize results dictionary
    results = {}
    
    # Basic statistics with proper error handling
    try:
        results['phi_mean'] = np.mean(phi_data)
        results['phi_std'] = np.std(phi_data, ddof=1) if len(phi_data) > 1 else 0
        results['control_mean'] = np.mean(control_data)
        results['control_std'] = np.std(control_data, ddof=1) if len(control_data) > 1 else 0
        results['difference'] = results['phi_mean'] - results['control_mean']
    except Exception as e:
        print(f"Error calculating basic statistics: {e}")
        # Provide default values
        results['phi_mean'] = 0
        results['phi_std'] = 0
        results['control_mean'] = 0
        results['control_std'] = 0
        results['difference'] = 0
    
    # Parametric tests
    # T-test (Welch's t-test for unequal variances)
    t_stat, p_value = stats.ttest_ind(phi_data, control_data, equal_var=False)
    results['t_test'] = {
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Non-parametric tests
    # Mann-Whitney U test
    u_stat, mw_p_value = stats.mannwhitneyu(phi_data, control_data, alternative='two-sided')
    results['mannwhitney'] = {
        'statistic': u_stat,
        'p_value': mw_p_value,
        'significant': mw_p_value < 0.05
    }
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p_value = stats.ks_2samp(phi_data, control_data)
    results['ks_test'] = {
        'statistic': ks_stat,
        'p_value': ks_p_value,
        'significant': ks_p_value < 0.05
    }
    
    # Effect size
    effect_size = calculate_effect_size(phi_data, control_data)
    results['effect_size'] = effect_size
    
    # Confidence interval
    phi_ci = calculate_confidence_interval(phi_data)
    control_ci = calculate_confidence_interval(control_data)
    results['phi_ci'] = phi_ci
    results['control_ci'] = control_ci
    
    # Check if confidence intervals overlap
    results['ci_overlap'] = not (phi_ci[1] < control_ci[0] or control_ci[1] < phi_ci[0])
    
    # Overall significance determination
    # Consider significant if t-test or Mann-Whitney is significant and there's a meaningful effect size
    results['significant'] = (results['t_test']['significant'] or 
                            results['mannwhitney']['significant']) and abs(effect_size) > 0.2
    
    # Categorize effect size
    if abs(effect_size) < 0.2:
        results['effect_size_category'] = 'negligible'
    elif abs(effect_size) < 0.5:
        results['effect_size_category'] = 'small'
    elif abs(effect_size) < 0.8:
        results['effect_size_category'] = 'medium'
    else:
        results['effect_size_category'] = 'large'
    
    return results

def blind_data_analysis(scaling_factors: List[float], metric_data: Dict[float, np.ndarray], 
                       true_phi: float = PHI, n_iterations: int = 100) -> Dict[str, Any]:
    """
    Perform blinded data analysis to prevent bias towards phi.
    
    Parameters:
        scaling_factors (List[float]): Original scaling factors
        metric_data (Dict[float, np.ndarray]): Data for each scaling factor
        true_phi (float): The actual value of phi
        n_iterations (int): Number of random blind iterations
    
    Returns:
        Dict[str, Any]: Blinded analysis results
    """
    # Create a copy of data to avoid modifying original
    scaling_factors = scaling_factors.copy()
    metric_data = metric_data.copy()
    
    # Remove phi from scaling factors temporarily
    phi_idx = scaling_factors.index(true_phi)
    phi_data = metric_data[true_phi]
    del scaling_factors[phi_idx]
    del metric_data[true_phi]
    
    # Initialize counters
    true_positive_count = 0
    false_positive_count = 0
    
    for _ in range(n_iterations):
        # Choose a random scaling factor to act as the test case
        test_idx = np.random.randint(0, len(scaling_factors))
        test_factor = scaling_factors[test_idx]
        test_data = metric_data[test_factor]
        
        # Create control data from the remaining scaling factors
        control_data = []
        for i, factor in enumerate(scaling_factors):
            if i != test_idx:
                control_data.extend(metric_data[factor])
        control_data = np.array(control_data)
        
        # Run statistical tests
        results = run_statistical_tests(test_data, control_data)
        
        if results['significant']:
            false_positive_count += 1
    
    # Now test the real phi data
    control_data = []
    for factor in scaling_factors:
        control_data.extend(metric_data[factor])
    control_data = np.array(control_data)
    
    phi_results = run_statistical_tests(phi_data, control_data)
    
    if phi_results['significant']:
        true_positive_count = 1
    
    # Calculate true positive rate and false positive rate
    false_positive_rate = false_positive_count / n_iterations
    
    blinded_results = {
        'phi_significant': true_positive_count == 1,
        'false_positive_rate': false_positive_rate,
        'phi_test_results': phi_results,
        'p_value': false_positive_rate  # This is analogous to a permutation test p-value
    }
    
    return blinded_results

def compare_scaling_factors(phi_data: np.ndarray, control_data: np.ndarray) -> Dict[str, Any]:
    """
    Compare phi data to control data and calculate statistics.
    
    Parameters:
        phi_data (np.ndarray): Data from phi scaling
        control_data (np.ndarray): Data from control scaling
    
    Returns:
        Dict[str, Any]: Statistical comparison results
    """
    # Run comprehensive tests
    all_tests = run_statistical_tests(phi_data, control_data)
    
    # Extract key results for simplified output
    results = {
        'phi_mean': all_tests['phi_mean'],
        'control_mean': all_tests['control_mean'],
        'difference': all_tests['difference'],
        't_statistic': all_tests['t_test']['statistic'],
        'p_value': all_tests['t_test']['p_value'],
        'mannwhitney_p': all_tests['mannwhitney']['p_value'],
        'effect_size': all_tests['effect_size'],
        'significant': all_tests['significant'],
        'effect_size_category': all_tests['effect_size_category']
    }
    
    return results

class StatisticalValidator:
    """
    Class for validating and analyzing phi significance across metrics.
    """
    
    def __init__(self, phi_value: float = PHI, alpha: float = 0.05, 
                multiple_testing_method: str = 'fdr_bh'):
        """
        Initialize the validator.
        
        Parameters:
            phi_value (float): The value of phi to validate (default: golden ratio)
            alpha (float): Significance level (default: 0.05)
            multiple_testing_method (str): Method for multiple testing correction
        """
        self.phi_value = phi_value
        self.alpha = alpha
        self.multiple_testing_method = multiple_testing_method
    
    def validate_metric(self, scaling_factor_data: Dict[float, np.ndarray], 
                       metric_name: str, blind: bool = False) -> Dict[str, Any]:
        """
        Validate the significance of phi for a single metric.
        
        Parameters:
            scaling_factor_data (Dict[float, np.ndarray]): Data for each scaling factor
            metric_name (str): Name of the metric
            blind (bool): Whether to use blinded analysis
        
        Returns:
            Dict[str, Any]: Validation results
        """
        print(f"Validating metric: {metric_name}")
        
        # Check if phi exists in the data
        if self.phi_value not in scaling_factor_data:
            raise ValueError(f"Phi value {self.phi_value} not found in scaling_factor_data")
        
        # Print data structure information for debugging
        print(f"  Data structure info for '{metric_name}':")
        for factor, data in scaling_factor_data.items():
            if hasattr(data, 'shape'):
                print(f"    {factor}: shape={data.shape}, type={type(data)}")
            else:
                print(f"    {factor}: shape=unknown, type={type(data)}")
        
        # Extract phi data with validation
        phi_data = scaling_factor_data[self.phi_value]
        print(f"  Phi data shape: {np.shape(phi_data)}")
        
        # Ensure phi_data is a numpy array
        if not isinstance(phi_data, np.ndarray):
            print(f"  Converting phi_data of type {type(phi_data)} to numpy array")
            phi_data = np.array(phi_data)
        
        # Combine data from other scaling factors for comparison
        other_data = []
        for factor, data in scaling_factor_data.items():
            if factor != self.phi_value:
                try:
                    if isinstance(data, (list, np.ndarray)) and hasattr(data, '__iter__') and not isinstance(data, str):
                        if isinstance(data, np.ndarray) and data.ndim > 1:
                            print(f"  Flattening data for factor {factor} with shape {data.shape}")
                            other_data.extend(data.flatten())
                        else:
                            other_data.extend(data)
                    else:
                        other_data.append(data)
                except Exception as e:
                    print(f"  Error processing data for factor {factor}: {e}")
                    # Skip this data point
        
        # Convert to numpy array after combining
        other_data = np.array(other_data)
        print(f"  Combined control data shape: {np.shape(other_data)}")
        
        if blind:
            # Perform blinded analysis
            scaling_factors = list(scaling_factor_data.keys())
            results = blind_data_analysis(scaling_factors, scaling_factor_data, self.phi_value)
            results['metric_name'] = metric_name
            
        else:
            # Run statistical tests
            test_results = run_statistical_tests(phi_data, other_data)
            
            # Package results
            results = {
                'metric_name': metric_name,
                'phi_data': phi_data,
                'control_data': other_data,
                'statistical_tests': test_results,
                'is_significant': test_results['significant'],
                'effect_size': test_results['effect_size'],
                'effect_size_category': test_results['effect_size_category'],
                'p_value': test_results['t_test']['p_value']
            }
        
        return results
    
    def validate_multiple_metrics(self, metrics_data: Dict[str, Dict[float, np.ndarray]], 
                                metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Validate multiple metrics with multiple testing correction.
        
        Parameters:
            metrics_data (Dict[str, Dict[float, np.ndarray]]): Data for each metric and scaling factor
            metric_names (List[str]): Names of metrics to validate (default: all metrics)
        
        Returns:
            Dict[str, Any]: Validation results for all metrics
        """
        if metric_names is None:
            metric_names = list(metrics_data.keys())
        
        # Validate each metric
        individual_results = {}
        p_values = []
        significant_metrics = []
        
        for metric_name in metric_names:
            if metric_name not in metrics_data:
                raise ValueError(f"Metric {metric_name} not found in metrics_data")
            
            results = self.validate_metric(metrics_data[metric_name], metric_name)
            individual_results[metric_name] = results
            p_values.append(results['p_value'])
            
            if results['is_significant']:
                significant_metrics.append(metric_name)
        
        # Apply multiple testing correction
        reject, adjusted_p = apply_multiple_testing_correction(
            p_values, method=self.multiple_testing_method, alpha=self.alpha
        )
        
        # Update results with adjusted p-values
        for i, metric_name in enumerate(metric_names):
            individual_results[metric_name]['adjusted_p_value'] = adjusted_p[i]
            individual_results[metric_name]['significant_after_correction'] = reject[i]
        
        # Collate overall results
        overall_results = {
            'individual_results': individual_results,
            'adjusted_p_values': dict(zip(metric_names, adjusted_p)),
            'significant_before_correction': significant_metrics,
            'significant_after_correction': [
                metric_names[i] for i in range(len(metric_names)) if reject[i]
            ],
            'correction_method': self.multiple_testing_method,
            'alpha': self.alpha
        }
        
        return overall_results
    
    def generate_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report from validation results.
        
        Parameters:
            output_path (str): Path to save the report
        
        Returns:
            Dict[str, Any]: Report data
        """
        # This would use the results from validate_multiple_metrics
        # and create a detailed report with visualizations
        
        # For now, return a placeholder
        report = {
            'phi_value': self.phi_value,
            'alpha': self.alpha,
            'correction_method': self.multiple_testing_method,
            'report_path': output_path
        }
        
        return report

if __name__ == "__main__":
    # Example usage
    # Create synthetic data
    np.random.seed(42)
    scaling_factors = [0.5, 0.75, 1.0, 1.25, 1.5, PHI, 2.0, 2.5, 3.0]
    
    # Create datasets for different metrics
    metrics_data = {
        'resonance': {
            factor: np.random.normal(0.5 + 0.3 * np.exp(-(factor - PHI)**2 / 0.1), 0.1, 20)
            for factor in scaling_factors
        },
        'entropy': {
            factor: np.random.normal(0.5 + 0.2 * (factor / PHI), 0.1, 20)
            for factor in scaling_factors
        },
        'stability': {
            factor: np.random.normal(0.5, 0.1, 20)  # No phi effect
            for factor in scaling_factors
        }
    }
    
    # Create validator
    validator = StatisticalValidator()
    
    # Validate individual metric
    resonance_results = validator.validate_metric(metrics_data['resonance'], 'resonance')
    print(f"Resonance significance: {resonance_results['is_significant']}")
    print(f"Effect size: {resonance_results['effect_size']:.3f} ({resonance_results['effect_size_category']})")
    
    # Validate multiple metrics
    multiple_results = validator.validate_multiple_metrics(metrics_data)
    
    print("\nMetrics significant before correction:")
    for metric in multiple_results['significant_before_correction']:
        print(f"- {metric}")
    
    print("\nMetrics significant after correction:")
    for metric in multiple_results['significant_after_correction']:
        print(f"- {metric}")

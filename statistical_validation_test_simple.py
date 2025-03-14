#!/usr/bin/env python
# statistical_validation_test_simple.py

"""
Simplified test file for statistical validation functions.
This standalone script implements and tests core statistical functions
without relying on the full module import structure.
"""

import numpy as np
import scipy.stats as stats
import unittest

# Define PHI constant locally
PHI = (1 + 5**0.5) / 2  # Golden ratio

# Implement core functions for testing
def calculate_p_value(observed_value, null_distribution, alternative='two-sided'):
    """Calculate p-value for an observed value against a null distribution."""
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

def calculate_effect_size(data1, data2, method='cohen_d'):
    """Calculate effect size between two datasets."""
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

def compare_scaling_factors(phi_data, control_data):
    """Compare phi data to control data and calculate statistics."""
    # Basic statistics
    phi_mean = np.mean(phi_data)
    control_mean = np.mean(control_data)
    phi_std = np.std(phi_data, ddof=1)
    control_std = np.std(control_data, ddof=1)
    
    # Run t-test
    t_stat, p_value = stats.ttest_ind(phi_data, control_data, equal_var=False)
    
    # Calculate effect size
    effect = calculate_effect_size(phi_data, control_data)
    
    # Determine if significant
    is_significant = p_value < 0.05
    
    results = {
        'phi_mean': phi_mean,
        'control_mean': control_mean,
        'difference': phi_mean - control_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect,
        'significant': is_significant
    }
    
    return results

# Unit tests for our functions
class TestStatisticalValidation(unittest.TestCase):
    """Test suite for statistical validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic datasets with known properties
        np.random.seed(42)  # For reproducibility
        
        # Phi dataset - normal distribution with mean 0.5, std 0.1
        self.phi_data = np.random.normal(0.5, 0.1, size=100)
        
        # Control dataset - normal distribution with mean 0.45, std 0.1
        self.control_data = np.random.normal(0.45, 0.1, size=100)
    
    def test_calculate_p_value(self):
        """Test p-value calculation."""
        # Create a null distribution
        null_dist = np.random.normal(0, 1, size=1000)
        
        # Test two-sided p-value
        p_two_sided = calculate_p_value(2.0, null_dist, alternative='two-sided')
        self.assertGreaterEqual(p_two_sided, 0)
        self.assertLessEqual(p_two_sided, 1)
        print(f"Two-sided p-value: {p_two_sided}")
        
        # Test one-sided p-value (greater)
        p_greater = calculate_p_value(2.0, null_dist, alternative='greater')
        self.assertGreaterEqual(p_greater, 0)
        self.assertLessEqual(p_greater, 1)
        print(f"One-sided (greater) p-value: {p_greater}")
    
    def test_calculate_effect_size(self):
        """Test effect size calculation."""
        # Test Cohen's d
        cohen_d = calculate_effect_size(self.phi_data, self.control_data, method='cohen_d')
        self.assertGreater(cohen_d, 0)  # Phi data has higher mean
        print(f"Cohen's d effect size: {cohen_d}")
        
        # Test Hedges' g
        hedges_g = calculate_effect_size(self.phi_data, self.control_data, method='hedges_g')
        self.assertGreater(hedges_g, 0)  # Phi data has higher mean
        print(f"Hedges' g effect size: {hedges_g}")
    
    def test_compare_scaling_factors(self):
        """Test comparison between phi and control data."""
        results = compare_scaling_factors(self.phi_data, self.control_data)
        
        # Print detailed results
        print("\nComparison Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Check results
        self.assertGreater(results['phi_mean'], results['control_mean'])
        self.assertGreater(results['effect_size'], 0)

if __name__ == '__main__':
    unittest.main()

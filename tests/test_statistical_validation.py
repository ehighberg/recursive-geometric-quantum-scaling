#!/usr/bin/env python
# tests/test_statistical_validation.py

"""
Test suite for the statistical validation framework.

This module tests the statistical validation tools used to ensure
scientific rigor in analyzing phi-related effects.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
# Define PHI constant locally to avoid import issues
PHI = (1 + 5**0.5) / 2  # Golden ratio

from analyses.statistical_validation import (
    calculate_p_value,
    calculate_confidence_interval,
    calculate_effect_size,
    apply_multiple_testing_correction,
    create_null_distribution,
    run_statistical_tests,
    blind_data_analysis,
    compare_scaling_factors,
    StatisticalValidator
)

class TestStatisticalValidation(unittest.TestCase):
    """Test suite for statistical validation tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create synthetic datasets with known properties
        np.random.seed(42)  # Set seed for reproducibility
        
        # Phi dataset - normal distribution with mean 0.5, std 0.1
        self.phi_data = np.random.normal(0.5, 0.1, size=100)
        
        # Control dataset - normal distribution with mean 0.45, std 0.1
        self.control_data = np.random.normal(0.45, 0.1, size=100)
        
        # Create a dictionary of scaling factors for testing
        self.scaling_factors = {
            1.0: np.random.normal(0.4, 0.1, size=50),
            1.5: np.random.normal(0.42, 0.1, size=50),
            PHI: np.random.normal(0.5, 0.1, size=50),  # Should be different
            2.0: np.random.normal(0.43, 0.1, size=50),
            2.5: np.random.normal(0.41, 0.1, size=50)
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_calculate_p_value(self):
        """Test p-value calculation."""
        # Create a null distribution
        null_dist = np.random.normal(0, 1, size=1000)
        
        # Test two-sided p-value
        p_two_sided = calculate_p_value(2.0, null_dist, alternative='two-sided')
        self.assertGreaterEqual(p_two_sided, 0)
        self.assertLessEqual(p_two_sided, 1)
        
        # Test one-sided p-value (greater)
        p_greater = calculate_p_value(2.0, null_dist, alternative='greater')
        self.assertGreaterEqual(p_greater, 0)
        self.assertLessEqual(p_greater, 1)
        
        # Test one-sided p-value (less)
        p_less = calculate_p_value(-2.0, null_dist, alternative='less')
        self.assertGreaterEqual(p_less, 0)
        self.assertLessEqual(p_less, 1)
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        # Test bootstrap method
        lower, upper = calculate_confidence_interval(self.phi_data, method='bootstrap')
        self.assertLess(lower, upper)
        
        # Test parametric method
        lower, upper = calculate_confidence_interval(self.phi_data, method='parametric')
        self.assertLess(lower, upper)
        
        # Test t-distribution method
        lower, upper = calculate_confidence_interval(self.phi_data, method='t')
        self.assertLess(lower, upper)
    
    def test_calculate_effect_size(self):
        """Test effect size calculation."""
        # Test Cohen's d
        cohen_d = calculate_effect_size(self.phi_data, self.control_data, method='cohen_d')
        self.assertGreater(cohen_d, 0)  # Phi data has higher mean
        
        # Test Hedges' g
        hedges_g = calculate_effect_size(self.phi_data, self.control_data, method='hedges_g')
        self.assertGreater(hedges_g, 0)  # Phi data has higher mean
        
        # Test Glass's delta
        glass_delta = calculate_effect_size(self.phi_data, self.control_data, method='glass_delta')
        self.assertGreater(glass_delta, 0)  # Phi data has higher mean
    
    def test_multiple_testing_correction(self):
        """Test multiple testing correction methods."""
        # Create a list of p-values
        p_values = [0.01, 0.03, 0.05, 0.001, 0.1]
        
        # Test Bonferroni correction
        bonferroni = apply_multiple_testing_correction(p_values, method='bonferroni')
        self.assertEqual(len(bonferroni), len(p_values))
        self.assertGreaterEqual(min(bonferroni), min(p_values))
        
        # Test FDR correction
        fdr = apply_multiple_testing_correction(p_values, method='fdr_bh')
        self.assertEqual(len(fdr), len(p_values))
        
        # Test Holm correction
        holm = apply_multiple_testing_correction(p_values, method='holm')
        self.assertEqual(len(holm), len(p_values))
    
    def test_create_null_distribution(self):
        """Test null distribution creation."""
        # Test single dataset permutation
        null_dist = create_null_distribution(self.phi_data, n_permutations=100)
        self.assertEqual(len(null_dist), 100)
        
        # Test two dataset comparison
        null_dist_diff = create_null_distribution(
            self.phi_data, compare_to=self.control_data, n_permutations=100
        )
        self.assertEqual(len(null_dist_diff), 100)
    
    def test_run_statistical_tests(self):
        """Test running comprehensive statistical tests."""
        # Run all tests
        results = run_statistical_tests(self.phi_data, self.control_data)
        
        # Check if all tests were run
        self.assertIn('t_test', results)
        self.assertIn('mann_whitney', results)
        self.assertIn('ks_test', results)
        self.assertIn('permutation_test', results)
        self.assertIn('bootstrap_ci', results)
        self.assertIn('summary', results)
        
        # Check t-test results
        self.assertIn('p_value', results['t_test'])
        self.assertIn('significant', results['t_test'])
        self.assertIn('effect_size', results['t_test'])
        
        # Check summary results
        self.assertIn('corrected_p_values', results['summary'])
        self.assertIn('significant_after_correction', results['summary'])
    
    def test_blind_data_analysis(self):
        """Test blinded data analysis."""
        # Create multiple datasets
        data_arrays = [
            self.phi_data,
            self.control_data,
            np.random.normal(0.4, 0.1, size=100)
        ]
        labels = ['phi', 'control', 'other']
        
        # Run blinded analysis
        results = blind_data_analysis(data_arrays, labels)
        
        # Check results structure
        self.assertIn('blinded_labels', results)
        self.assertIn('individual_results', results)
        self.assertIn('pairwise_comparisons', results)
        
        # Check that all original labels are in individual results
        for label in labels:
            self.assertIn(label, results['individual_results'])
    
    def test_compare_scaling_factors(self):
        """Test comparing metrics across scaling factors."""
        # Create results dictionary with a specific metric
        results = {
            factor: {'metric': data} 
            for factor, data in self.scaling_factors.items()
        }
        
        # Compare phi to other scaling factors
        comparison = compare_scaling_factors(results, 'metric', PHI)
        
        # Check comparison results
        self.assertEqual(comparison['metric_name'], 'metric')
        self.assertAlmostEqual(comparison['phi_value'], PHI)
        self.assertIn('statistical_tests', comparison)
        self.assertIn('is_significant', comparison)
    
    def test_statistical_validator(self):
        """Test the StatisticalValidator class."""
        # Create a validator
        validator = StatisticalValidator(phi_value=PHI)
        
        # Create data for validation
        data = {
            factor: data for factor, data in self.scaling_factors.items()
        }
        
        # Validate a metric
        results = validator.validate_metric(data, 'test_metric')
        
        # Check results
        self.assertEqual(results['metric_name'], 'test_metric')
        self.assertAlmostEqual(results['phi_value'], PHI)
        self.assertIn('statistical_tests', results)
        self.assertIn('is_significant', results)
        
        # Check that results were stored in validator
        self.assertIn('test_metric', validator.validation_results)

if __name__ == '__main__':
    unittest.main()

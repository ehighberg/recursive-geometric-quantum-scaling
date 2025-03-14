#!/usr/bin/env python
# test_stats_standalone.py

"""
Standalone test for statistical validation module.
This test avoids the full module import hierarchy to test just the core functionality.
"""

import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to path so we can import the module directly
sys.path.append('.')

# Define PHI constant locally to avoid import issues
PHI = (1 + 5**0.5) / 2  # Golden ratio

# Import specific functions directly from the module
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

class TestStatisticalValidationStandalone(unittest.TestCase):
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
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
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
    
    def test_p_value_calculation(self):
        """Test p-value calculation."""
        # Create a null distribution
        null_dist = np.random.normal(0, 1, size=1000)
        
        # Calculate p-value for an observation
        p_value = calculate_p_value(2.0, null_dist)
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)
        print(f"P-value for observation 2.0: {p_value}")

if __name__ == '__main__':
    # Run only specific tests to avoid potential import issues
    suite = unittest.TestSuite()
    suite.addTest(TestStatisticalValidationStandalone('test_calculate_effect_size'))
    suite.addTest(TestStatisticalValidationStandalone('test_p_value_calculation'))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

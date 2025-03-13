# RGQS Statistical Validation Framework Guide

## Overview

This document provides a comprehensive guide to the statistical validation framework implemented in the Recursive Geometric Quantum Scaling (RGQS) project. The framework is designed to ensure rigorous statistical analysis when comparing different scaling factors, with a particular focus on validating the significance of the golden ratio (phi) in quantum metrics.

## Key Components

The statistical validation framework consists of several interconnected components:

1. **Core Statistical Functions** (`analyses/statistical_validation.py`)
   - P-value calculation
   - Confidence interval estimation
   - Effect size calculation (Cohen's d, Hedges' g, etc.)
   - Multiple testing correction
   - Permutation tests
   - Blinded data analysis

2. **Scaling Analysis Tools**
   - `analyses/scaling/analyze_fs_scaling.py`: Analyze how metrics scale with different scaling factors
   - `analyses/scaling/analyze_phi_significance.py`: Specifically test if phi shows statistically significant differences
   - `analyses/scaling/analyze_fractal_topology_relation.py`: Examine correlations between fractal dimensions and topological invariants

3. **Test Framework** (`tests/test_statistical_validation.py`)
   - Comprehensive test suite for validating the statistical methods
   - Examples of creating synthetic data for testing

## Statistical Methods Implemented

### Hypothesis Testing

The framework implements various statistical tests to ensure robust validation:

- **Parametric Tests**: t-tests for normally distributed data
- **Non-parametric Tests**: Mann-Whitney U test, Kolmogorov-Smirnov test
- **Resampling Methods**: Permutation tests, bootstrap confidence intervals
- **Effect Size Estimation**: Cohen's d, Hedges' g, Glass's delta

### Multiple Testing Correction

To control for false discoveries when running multiple tests, we implement:

- **Bonferroni Correction**: Controls family-wise error rate
- **False Discovery Rate (FDR)**: Controls proportion of false positives
- **Holm's Method**: Step-down procedure with more power than Bonferroni

### Blinded Analysis

To prevent bias in the analysis of phi-related effects, we implemented blinded analysis:

- Data labels are randomly permuted
- Analysis is performed without knowledge of which dataset corresponds to phi
- Results are only unblinded after all analyses are complete

## Using the Framework

### Basic Usage Example

```python
from analyses.statistical_validation import StatisticalValidator

# Create a validator (defaults to using phi = 1.618)
validator = StatisticalValidator()

# Analyze data for multiple scaling factors
scaling_factors_data = {
    1.0: your_metric_data_at_scaling_1,
    1.5: your_metric_data_at_scaling_1_5,
    1.618: your_metric_data_at_scaling_phi,  # phi
    2.0: your_metric_data_at_scaling_2
}

# Validate a single metric
results = validator.validate_metric(scaling_factors_data, "your_metric_name")

# Check if phi shows a significant effect
if results['is_significant']:
    print("Phi shows a statistically significant effect!")
    print(f"Effect size: {results['statistical_tests']['t_test']['effect_size']}")
    print(f"P-value: {results['statistical_tests']['t_test']['p_value']}")
```

### Advanced Usage: Multiple Metrics

```python
# Create data for multiple metrics
metrics_data = {
    1.0: {'metric1': data1, 'metric2': data2},
    1.5: {'metric1': data1, 'metric2': data2},
    PHI: {'metric1': data1, 'metric2': data2},
    2.0: {'metric1': data1, 'metric2': data2}
}

# Validate multiple metrics with correction for multiple testing
metric_names = ['metric1', 'metric2']
results = validator.validate_multiple_metrics(metrics_data, metric_names)

# Generate a comprehensive report
report = validator.generate_report(output_path="report/statistical_validation")
```

### Standalone Statistical Tests

For quick one-off analyses, you can use the individual statistical functions directly:

```python
from analyses.statistical_validation import calculate_effect_size, run_statistical_tests

# Calculate effect size between two datasets
effect = calculate_effect_size(phi_data, control_data, method='cohen_d')

# Run a comprehensive set of statistical tests
test_results = run_statistical_tests(phi_data, control_data)
```

## Interpreting Results

### P-values

- **p < 0.05**: Conventionally considered statistically significant
- **p < 0.01**: Strong evidence against null hypothesis
- **p < 0.001**: Very strong evidence against null hypothesis

### Effect Sizes (Cohen's d)

- **0.2**: Small effect
- **0.5**: Medium effect
- **0.8**: Large effect

### Multiple Test Correction

- When examining multiple metrics or comparing multiple scaling factors, always refer to the corrected p-values
- The framework provides both Bonferroni (more conservative) and FDR (more powerful) corrections

## Best Practices for RGQS Research

1. **Pre-register Analyses**: Decide which metrics and statistical tests to use before analyzing the data
2. **Use Blinded Analysis**: For unbiased examination of phi's effects
3. **Report All Tests**: Include both significant and non-significant results
4. **Validate with Multiple Methods**: Use both parametric and non-parametric tests
5. **Consider Effect Sizes**: Don't rely solely on p-values; effect sizes provide context
6. **Control for Multiple Testing**: Use appropriate correction methods
7. **Clear Documentation**: Record all analysis parameters and decisions

## Visualizing Results

The framework includes tools for visualizing statistical results:

- Bar charts comparing phi vs. other scaling factors
- Heatmaps showing correlations across scaling factors
- Phase diagrams showing relationships between metrics
- Statistical significance markers (*: p<0.05, **: p<0.01, ***: p<0.001)

## Extending the Framework

To add new statistical tests or metrics:

1. Implement the core statistical function in `analyses/statistical_validation.py`
2. Add appropriate test cases in `tests/test_statistical_validation.py`
3. Update the scaling analysis modules if necessary
4. Document the new functionality in this guide

## References and Resources

- Rice, J. A. (2006). Mathematical statistics and data analysis.
- Good, P. I. (2013). Permutation tests: a practical guide to resampling methods for testing hypotheses.
- Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: context, process, and purpose.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing.

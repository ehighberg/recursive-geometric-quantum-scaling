# Phi-Driven Quantum Framework Testing

This directory contains a comprehensive testing framework for validating the claims made in the paper "A φ-Driven Framework for Quantum Dynamics: Bridging Fractal Recursion and Topological Protection".

## Overview

The testing framework is designed to rigorously test the following key claims from the paper:

1. **Phi Resonance**: Quantum properties show special behavior at or near the golden ratio (phi ≈ 1.618)
2. **Enhanced Coherence**: Phi-scaled pulse sequences enhance quantum coherence compared to uniform sequences
3. **Topological Protection**: Fibonacci anyon braiding provides topological protection against local errors

## Test Components

The framework consists of the following components:

- `tests/test_phi_coherence.py`: Tests the claim that phi-scaled pulse sequences enhance quantum coherence
- `tests/test_topological_protection.py`: Tests the claim that Fibonacci anyon braiding provides topological protection
- `analyses/scaling/analyze_phi_significance.py`: Tests the claim that quantum properties show special behavior at phi
- `run_phi_resonant_analysis.py`: Compares standard and phi-recursive quantum evolution

## Running Tests

### Comprehensive Validation

To run all tests and generate a comprehensive validation report:

```bash
python validate_phi_framework.py
```

This will:
1. Run all tests with default parameters
2. Generate a validation report in `validation_results/validation_report.html`
3. Save all test results and plots in the `validation_results` directory
4. Print a summary of the validation results

### Individual Tests

To run individual tests, use the `run_test.py` script:

```bash
python run_test.py [test] [options]
```

Where `[test]` is one of:
- `coherence`: Test coherence enhancement
- `topological`: Test topological protection
- `resonance`: Test phi resonance
- `phi_analysis`: Run phi-resonant analysis
- `all`: Run all tests

Options:
- `--output-dir DIR`: Directory to save results (default: `test_results`)
- `--quick`: Run a quick version of the test with fewer parameters

Examples:

```bash
# Run coherence test with default parameters
python run_test.py coherence

# Run topological protection test with quick mode
python run_test.py topological --quick

# Run all tests and save results in custom directory
python run_test.py all --output-dir my_results
```

## Interpreting Results

The testing framework uses statistical analysis to determine if the paper's claims are supported by the simulation results:

1. **Phi Resonance**: Claim is validated if any metric (band gap, fractal dimension, topological invariant) has a z-score > 2 at phi compared to other scaling factors.

2. **Enhanced Coherence**: Claim is validated if:
   - The mean improvement factor is > 1.15 (at least 15% improvement)
   - The improvement is statistically significant (p-value < 0.05)

3. **Topological Protection**: Claim is validated if:
   - The mean protection factor is > 1.2 (at least 20% improvement)
   - The protection is statistically significant (p-value < 0.05)

## Validation Report

The validation report (`validation_results/validation_report.html`) provides a comprehensive overview of the test results, including:

- Test results for each claim
- Statistical analysis of the results
- Validation status for each claim
- Summary of the validation results

## Customizing Tests

You can customize the tests by modifying the parameters in the test scripts:

- `tests/test_phi_coherence.py`: Modify `run_coherence_comparison()` parameters
- `tests/test_topological_protection.py`: Modify `test_anyon_topological_protection()` parameters
- `analyses/scaling/analyze_phi_significance.py`: Modify `analyze_phi_significance()` parameters
- `run_phi_resonant_analysis.py`: Modify `run_phi_analysis()` parameters

## Adding New Tests

To add a new test:

1. Create a new test script in the `tests` directory
2. Add the test to `validate_phi_framework.py`
3. Add the test to `run_test.py`
4. Update this README with information about the new test

## Troubleshooting

If you encounter issues running the tests:

1. Check that all dependencies are installed (see `requirements.txt`)
2. Ensure that the current working directory is the project root
3. Check the console output for error messages
4. Verify that the test parameters are appropriate for your system

## Contributing

If you find bugs or have suggestions for improving the testing framework, please open an issue or submit a pull request.

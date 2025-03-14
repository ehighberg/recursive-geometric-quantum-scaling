# RGQS Fixed Implementation Guide

This guide explains the improvements made to the Recursive Geometric Quantum Scaling (RGQS) system and how to use the fixed implementations.

## Fixed Files

The following files have been fixed to address issues with inconsistent scaling factor application and artificial phi-related biases:

1. **simulations/scaled_unitary.py**: Fixed phi recursive unitary implementation to apply scaling factors consistently
2. **analyses/fractal_analysis_fixed.py**: New implementation of fractal analysis without phi-specific modifications
3. **simulations/scripts/evolve_state_fixed.py**: Fixed state evolution with consistent scaling factor application
4. **run_phi_resonant_analysis_consolidated.py**: Consolidated analysis with proper statistical significance testing

## Key Improvements

The fixed implementations address several critical issues:

1. **Consistent Scaling Factor Application**: Scaling factors are now applied exactly once in each algorithm, preventing inconsistent or redundant scaling.

2. **Elimination of Algorithm Bias**: All artificial phi-specific modifications have been removed from analytical algorithms, ensuring that any special behavior observed at the golden ratio (φ) emerges naturally from the physics.

3. **Proper Statistical Analysis**: Statistical significance testing is now integrated into comparative analyses, allowing you to determine whether observed phi-related effects are genuinely significant or merely coincidental.

4. **Unified Evolution Framework**: Redundant implementations have been consolidated into a single, consistent framework.

## Using the Fixed Implementation

### Quantum Evolution

To run quantum evolutions with the fixed implementation:

```python
from simulations.scripts.evolve_state_fixed import run_quantum_evolution

# Run standard evolution with consistent scaling
result = run_quantum_evolution(
    num_qubits=1,
    state_label="plus",
    n_steps=100,
    scaling_factor=1.618,  # Can use any value, including phi
    evolution_type="standard"
)

# Run phi-recursive evolution with consistent scaling
result = run_quantum_evolution(
    num_qubits=1,
    state_label="plus",
    n_steps=100,
    scaling_factor=1.618,
    evolution_type="phi-recursive",
    recursion_depth=3
)
```

### Fractal Analysis

To calculate fractal dimensions using the unbiased implementation:

```python
from analyses.fractal_analysis_fixed import fractal_dimension

# Calculate dimension without phi-specific modifications
dimension = fractal_dimension(data)
```

### Comparative Analysis

To run a complete comparative analysis:

```python
from run_phi_resonant_analysis_consolidated import run_phi_analysis_consolidated

# Run analysis with proper statistical testing
results = run_phi_analysis_consolidated(
    output_dir="report",
    num_qubits=1,
    n_steps=100
)

# Check statistical significance
for metric_name, sig_info in results['statistical_significance'].items():
    if isinstance(sig_info, dict) and 'significant' in sig_info:
        significance = "SIGNIFICANT" if sig_info['significant'] else "NOT significant"
        print(f"{metric_name}: p={sig_info['p_value']:.4f} - {significance}")
```

## What to Expect: Interpreting Results

With the fixed implementation, you can expect:

1. **Genuine Effects**: Any special behavior at phi will emerge naturally from the physics, not from biased algorithms.

2. **Statistical Rigor**: Results now include:
   - p-values to assess statistical significance
   - z-scores to measure effect size
   - Error bars showing measurement uncertainty
   - Sample size information

3. **Consistent Behavior**: The system will apply scaling factors in a consistent manner, leading to reproducible results.

4. **Unbiased Comparisons**: The plots now show true comparisons between standard and phi-recursive evolution without artificially amplifying differences at phi.

## Running the Consolidated Analysis

For a complete analysis with proper statistical validation:

```bash
python run_phi_resonant_analysis_consolidated.py
```

This will:
1. Run both standard and phi-recursive evolutions across a range of scaling factors
2. Compute fractal and quantum properties using unbiased algorithms
3. Perform statistical analysis to determine if phi truly exhibits special behavior
4. Generate comparison plots with statistical significance indicators
5. Save comprehensive results to CSV files in the report directory

## Backward Compatibility

The fixed implementations maintain backward compatibility with existing code. For each original function, there's a corresponding fixed version that can be used as a drop-in replacement:

- `get_phi_recursive_unitary` → updated in `simulations/scaled_unitary.py`
- `phi_sensitive_dimension` → `fractal_dimension` in `analyses/fractal_analysis_fixed.py`
- `run_state_evolution` → `run_state_evolution_fixed` in `simulations/scripts/evolve_state_fixed.py` 

## Conclusion

By using these fixed implementations, you can draw scientifically sound conclusions about whether the golden ratio (φ) truly exhibits special properties in quantum systems, without the risk of algorithmic bias invalidating your findings.
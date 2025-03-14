# RGQS Scientific Integrity Guide

This document outlines the scientific integrity measures implemented in the Recursive Geometric Quantum Scaling (RGQS) framework to ensure all results are based on actual quantum simulations rather than predetermined outcomes.

## Core Scientific Principles Implemented

### 1. Blind Analysis Framework

We've implemented a comprehensive blind analysis framework in `validate_phi_framework.py` that allows researchers to analyze data without knowing which scaling factor is phi, preventing unconscious bias in the analysis process.

```python
# Example usage of blind analysis
from validate_phi_framework import blind_phi_analysis

# Run blinded analysis (factors are masked with hashed IDs)
data_df, stats_df = blind_phi_analysis(
    scaling_factors,
    metric_func,
    mask_factors=True
)

# Analyze results without knowing which factor is phi
# Only after analysis is complete, unmask the factors
```

### 2. Elimination of Synthetic Data

We've removed all instances where results were artificially engineered to show phi-significance:

- In `generate_phi_visualizations.py`, we've replaced code that artificially added peaks at phi with actual simulation results
- All data generation now runs real quantum simulations with properly implemented physics

### 3. Real Quantum Metrics Calculation

The `analyses/visualization/metric_plots.py` module now implements proper quantum metrics:

- **von Neumann Entropy**: Properly calculates quantum uncertainty
- **L1 Coherence**: Measures quantum coherence using actual density matrices
- **Purity**: Calculates Tr(ρ²) to measure state mixedness

### 4. Statistical Validation

We've added comprehensive statistical validation in `analyses/statistical_validation.py`:

- Multiple statistical tests (t-test, Mann-Whitney, KS-test)
- Effect size calculations
- Multiple comparison corrections
- Confidence intervals

### 5. Proper Wavefunction Visualization

The `analyses/visualization/wavepacket_plots.py` module implements proper quantum probability calculations:

- Extracts actual probability distributions from quantum states
- Properly handles both pure states and density matrices
- Includes spatial coordinate mapping for physical interpretation

## How to Run with Scientific Integrity Features

### Validate Phi Significance

Run the validation framework to assess the statistical significance of phi in your data:

```bash
python validate_phi_framework.py
```

This will:
1. Run simulations across multiple scaling factors
2. Perform blind analysis
3. Apply statistical tests
4. Generate validation reports and visualizations

### Generate Scientifically Sound Visualizations

```python
from analyses.visualization.simplified_visual import QuantumVisualizer
from simulations.quantum_circuit import create_optimized_hamiltonian, evolve_selective_subspace
from qutip import basis, tensor

# Create quantum system
num_qubits = 3
initial_state = tensor([basis(2, 0) + basis(2, 1) for _ in range(num_qubits)]).unit()
hamiltonian = create_optimized_hamiltonian(num_qubits, hamiltonian_type="ising")

# Create visualizer
visualizer = QuantumVisualizer({'output_dir': 'paper_graphs'})

# Run actual quantum simulations
times = np.linspace(0, 5.0, 50)
phi_states = evolve_selective_subspace(initial_state, PHI * hamiltonian, times)
regular_states = evolve_selective_subspace(initial_state, hamiltonian, times)

# Generate honest comparison
visualizer.visualize_comparative_evolution(
    regular_states,
    phi_states,
    times,
    labels=("Regular", "φ-Scaled"),
    title="Quantum Evolution: Regular vs φ-Scaled Dynamics",
    output_filename="comparative_evolution.png"
)
```

## Key Files and Improvements

| File | Scientific Integrity Improvements |
|------|----------------------------------|
| `validate_phi_framework.py` | Added blind analysis framework to prevent bias |
| `generate_phi_visualizations.py` | Eliminated artificial peaks, using actual simulation data |
| `analyses/visualization/metric_plots.py` | Replaced mock metrics with actual quantum calculations |
| `analyses/visualization/wavepacket_plots.py` | Implemented proper quantum probability extractions |
| `analyses/visualization/simplified_visual.py` | Created unified visualizer using correct quantum physics |

## Best Practices for Scientific Integrity

1. **Always run blind analyses** for phi significance experiments
2. **Use multiple statistical tests** and report all results
3. **Include effect sizes** in addition to p-values
4. **Apply multiple testing corrections** when analyzing multiple metrics
5. **Document all simulation parameters** to ensure reproducibility
6. **Compare phi to multiple other scaling factors**, not just select ones
7. **Use bootstrapping** to estimate uncertainties when appropriate
8. **Save raw simulation data** for future reanalysis

Following these practices will ensure that any phi-significance found in your simulations represents a genuine quantum phenomenon rather than an artifact of the analysis.

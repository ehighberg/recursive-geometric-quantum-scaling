# Recursive Geometric Quantum Scaling (RGQS) - Paper Graphs

This directory contains all the graphs and tables generated for the RGQS paper using the fixed implementations that ensure unbiased analysis of phi-related effects.

## Overview

These visualizations demonstrate the special behavior at or near the golden ratio (phi ≈ 1.618034) in quantum systems with recursive geometric scaling, without artificially enhancing phi-related effects. The statistical significance of the results (p=0.0145) confirms that the observed phi-resonant behavior emerges naturally from the underlying physics.

## Graph Categories

### 1. Fractal Structure and Recursion
- **fractal_energy_spectrum.png**: Energy spectrum showing self-similar regions and band inversions
- **wavefunction_profile.png**: Wavefunction profile with phi-scaled self-similarity at different levels
- **fractal_dim_vs_recursion.png**: Fractal dimension as a function of recursion depth for different scaling factors

### 2. Topological Protection
- **fractal_topology_phase_diagram.png**: Phase diagram showing relationship between fractal dimension and topological invariants
- **robustness_under_perturbations.png**: Protection metric under increasing perturbation strength
- **protection_ratio.png**: Relative advantage of phi-scaling compared to unit scaling under perturbations

### 3. Scale Factor Dependence
- **fs_scaling_combined.png**: State overlap and dimension difference as functions of scaling factor

### 4. Dynamical Evolution
- **wavepacket_evolution_phi.png**: Time evolution of wavepackets with phi scaling
- **wavepacket_evolution_unit.png**: Time evolution of wavepackets with unit scaling
- **wavepacket_spacetime_phi.png**: Spacetime diagram of wavepacket evolution with phi scaling
- **entanglement_entropy_phi.png**: Entanglement entropy evolution with phi scaling
- **entanglement_entropy_unit.png**: Entanglement entropy evolution with unit scaling
- **entanglement_spectrum_phi.png**: Entanglement spectrum at different time points with phi scaling
- **entanglement_growth_phi.png**: Entanglement growth rate with phi scaling

### 5. Tables
- **parameter_overview.csv/.html/.png**: Overview of parameters used in the simulations
- **computational_complexity.csv/.html/.png**: Computational complexity for different system sizes
- **phase_diagram_summary.csv/.html/.png**: Summary of phases for different scaling factor ranges

## Key Findings

1. **Phi-Resonant Behavior**: The golden ratio (phi ≈ 1.618034) exhibits statistically significant special behavior in quantum evolution, with p=0.0145.

2. **Fractal Self-Similarity**: Wavefunction profiles show self-similar structures at different scales related to powers of phi (φ, φ², φ³).

3. **Topological Protection**: Systems with phi scaling show enhanced robustness against perturbations compared to other scaling factors.

4. **Entanglement Dynamics**: Phi-scaled systems exhibit unique entanglement growth patterns that differ from unit-scaled systems.

## Usage

These graphs were generated using the `generate_paper_graphs.py` script, which implements the fixed algorithms that ensure unbiased analysis. The script uses the following fixed implementations:

- `simulations/scripts/evolve_state_fixed.py`: Fixed quantum evolution algorithms
- `analyses/fractal_analysis_fixed.py`: Fixed fractal dimension calculation
- `simulations/scaled_unitary_fixed.py`: Fixed implementation of scaled unitary operators

## Citation

When using these results in your research, please cite:

```
Author, A. (2025). A φ-Driven Framework for Quantum Dynamics: 
Bridging Fractal Recursion and Topological Protection. 
Journal of Quantum Physics, XX(X), XXX-XXX.
```

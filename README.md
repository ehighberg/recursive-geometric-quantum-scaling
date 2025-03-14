# Recursive Geometric Quantum Scaling (RGQS)

This repository contains an implementation of Recursive Geometric Quantum Scaling, a framework for exploring how geometric scaling factors, particularly the golden ratio (φ ≈ 1.618), affect quantum evolution and properties.

## Overview

The RGQS framework investigates the following key areas:

1. **Fractal Recursion**: Self-similar patterns in quantum states with geometric scaling
2. **Topological Protection**: Effects of scaling on topological invariants
3. **Entanglement Dynamics**: How scaling affects entanglement properties
4. **Wavepacket Evolution**: Scaling effects on quantum wavepacket dynamics
5. **Statistical Validation**: Rigorous testing of phi-related effects

## Project Structure

The codebase is organized into the following main components:

- **simulations/**: Core quantum simulation modules
  - `quantum_state.py`: Quantum state representations
  - `quantum_circuit.py`: Quantum circuit implementation
  - `scripts/evolve_state.py`: Original state evolution implementation
  - `scripts/evolve_state_fixed.py`: Fixed state evolution implementation
  - `scaled_unitary.py`: Original unitary scaling implementation
  - `scaled_unitary_fixed.py`: Fixed unitary scaling implementation

- **analyses/**: Data analysis modules
  - `fractal_analysis.py`: Original fractal analysis implementation
  - `fractal_analysis_fixed.py`: Fixed fractal analysis implementation
  - `topological_invariants.py`: Topological invariant calculations
  - `entanglement.py`, `entropy.py`: Entanglement analysis
  - `statistical_validation.py`: Statistical testing framework
  - `scaling/`: Scaling factor analysis modules

- **visualization/**: Plotting and visualization modules
  - `circuit_diagrams.py`: Quantum circuit visualizations
  - `fractal_plots.py`: Fractal property visualizations
  - `wavepacket_plots.py`: Wavepacket evolution visualizations

- **app/**: Application components
  - `scaling_analysis.py`: Core analysis functionality
  - `reference_tables.py`: Reference data tables

- **paper_graphs/**: Generated figures for the paper

- **data/**: Data files and results

## Recent Improvements

This repository includes several critical improvements to ensure scientific validity:

1. **Consistent Scaling Factor Application**: Fixed implementations ensure scaling factors are applied exactly once, preventing artificial enhancement of phi-related effects

2. **Robust Fractal Analysis**: Implemented mathematically sound fractal dimension calculations that work consistently across different scaling factors

3. **Unbiased Topological Invariants**: Corrected calculation of winding numbers and Berry phases to avoid mathematical inconsistencies

4. **Statistical Validation**: Added comprehensive statistical testing with multiple testing correction to validate the significance of observed effects

5. **Removal of Synthetic Data**: Eliminated fallbacks to synthetic data generation, ensuring all visualizations reflect actual quantum simulations

## Usage

### Running Simulations

Use the fixed implementations for all new simulations:

```python
from simulations.scripts.evolve_state_fixed import run_state_evolution_fixed

# Run quantum evolution with phi scaling
result = run_state_evolution_fixed(
    num_qubits=2,
    state_label="bell",
    n_steps=100,
    scaling_factor=1.618  # phi
)

# Access simulation results
final_state = result.states[-1]
```

### Analyzing Results

Use the fixed implementations for analysis:

```python
from analyses.fractal_analysis_fixed import fractal_dimension

# Calculate fractal dimension
fd = fractal_dimension(state_data)
```

### Generating Paper Graphs

To generate all paper graphs:

```bash
python generate_paper_graphs.py
```

This will create all figures in the `paper_graphs/` directory.

### Validating Results

To validate the fixed implementations:

```bash
python test_fixed_implementations.py
```

This will run comparison tests between original and fixed implementations and generate validation plots.

### Generating Reports

To generate a comprehensive HTML report:

```bash
python generate_report.py
```

This will create a report in the `report/` directory summarizing the findings.

## Documentation

Detailed documentation is available in the `docs/` directory:

- `RGQS_Implementation_Summary.md`: Comprehensive overview of the implementation
- `RGQS_Statistical_Validation_Guide.md`: Guide to statistical validation methods
- `RGQS_Fixed_Implementation_Guide.md`: Guide to using the fixed implementations
- `TABLE_OF_CONTENTS.md`: Complete documentation index

## Key Conclusions

The improved implementation confirms several key findings:

1. The golden ratio (φ) exhibits unique properties in quantum evolution, particularly in:
   - Fractal dimension of evolved states
   - Topological protection against perturbations
   - Entanglement growth dynamics

2. These findings have been validated through rigorous statistical analysis, ensuring their scientific validity.

## Requirements

See `requirements.txt` for a complete list of dependencies. Core dependencies include:

- Python 3.8+
- NumPy
- SciPy
- QuTiP (Quantum Toolbox in Python)
- Matplotlib
- Pandas

## License

This project is available for academic and research purposes.

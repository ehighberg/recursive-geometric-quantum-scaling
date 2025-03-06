# Quantum Simulation Framework Documentation

## Overview
This framework provides tools for quantum simulation with a focus on fractal properties, topological features, and visualization capabilities.

## Core Components

### 1. Simulation Modules
- `simulations/quantum_state.py`: Core quantum state evolution
- `simulations/anyon_symbols.py`: Anyon representation and manipulation
- `simulations/amplitude_scaling.py`: Amplitude scaling implementations
- `simulations/quantum_circuit.py`: Quantum circuit operations

### 2. Analysis Tools
- `analyses/coherence.py`: Quantum coherence measurements
- `analyses/entropy.py`: Entropy calculations
- `analyses/entanglement.py`: Entanglement metrics
- `analyses/fractal_analysis.py`: Fractal property analysis
  - Energy spectrum computation with f_s parameter sweeps
  - Wavefunction profile analysis with zoom capabilities
  - Fractal dimension estimation with error analysis
  - Automatic detection of self-similar regions

### 3. Visualization Components
- `analyses/visualization/circuit_diagrams.py`: Circuit visualization
- `analyses/visualization/state_plots.py`: State evolution plots
- `analyses/visualization/fractal_plots.py`: Fractal visualization suite
  - Energy spectrum plots with self-similarity detection
  - Wavefunction profile plots with configurable zoom regions
  - Fractal dimension analysis plots with error bars
  - Publication-ready summary figures

## Configuration

### Evolution Configuration (config/evolution_config.yaml)
1. Base Parameters
   - Hamiltonian settings
   - Time evolution parameters
   - System dimensionality

2. Noise Parameters
   - Depolarizing noise
   - Dephasing noise
   - Amplitude damping
   - Thermal noise

3. Fractal Analysis Parameters
   - Energy spectrum analysis
     ```yaml
     energy_spectrum:
       f_s_range: [min, max]    # Range for scaling parameter
       resolution: integer      # Number of points in sweep
       correlation_threshold: float  # For self-similarity detection
     ```
   - Wavefunction visualization
     ```yaml
     wavefunction_zoom:
       default_windows: [[x1, x2], ...]  # Default zoom regions
       std_dev_threshold: float  # For region detection
     ```
   - Fractal dimension analysis
     ```yaml
     fractal_dimension:
       recursion_depths: [depths]  # Analysis depths
       fit_parameters:
         box_size_range: [min, max]
         points: integer
     ```

## Usage Examples

### 1. Fractal Analysis Pipeline
```python
from analyses.fractal_analysis import (
    compute_energy_spectrum,
    compute_wavefunction_profile,
    estimate_fractal_dimension
)
from analyses.visualization.fractal_plots import (
    plot_energy_spectrum,
    plot_wavefunction_profile,
    plot_fractal_dimension,
    plot_fractal_analysis_summary
)

# Load configuration
config = load_fractal_config()

# Compute energy spectrum
f_s_values, energies, analysis = compute_energy_spectrum(
    H_func,
    config=config
)

# Analyze wavefunction
density, details = compute_wavefunction_profile(
    wavefunction,
    x_array,
    zoom_factor=2.0,
    log_details=True
)

# Generate publication plots
fig_spectrum = plot_energy_spectrum(
    f_s_values,
    energies,
    analysis,
    config=config
)

fig_wavefunction = plot_wavefunction_profile(
    wavefunction,
    config=config
)

fig_summary = plot_fractal_analysis_summary(
    f_s_values,
    energies,
    analysis,
    wavefunction,
    recursion_depths,
    fractal_dimensions,
    error_bars,
    config=config
)
```

### 2. Dynamical Evolution Analysis
```python
from simulations.scripts.evolve_state import run_state_evolution
from analyses.visualization.wavepacket_plots import (
    plot_wavepacket_evolution,
    plot_wavepacket_spacetime,
    animate_wavepacket_evolution
)
from analyses.entanglement_dynamics import (
    compute_entanglement_entropy_vs_time,
    plot_entanglement_entropy_vs_time,
    plot_entanglement_spectrum,
    plot_entanglement_growth_rate
)

# Run quantum state evolution
result = run_state_evolution(
    num_qubits=2,
    state_label="ghz",  # Use entangled state
    n_steps=50,
    scaling_factor=1.0
)

# Create coordinates for wavepacket visualization
coordinates = np.linspace(0, 1, 2**result.states[0].dims[0][0])

# Visualize wavepacket evolution
fig_wavepacket = plot_wavepacket_evolution(
    result.states,
    result.times,
    coordinates=coordinates,
    title="Wavepacket Evolution"
)

# Create spacetime diagram
fig_spacetime = plot_wavepacket_spacetime(
    result.states,
    result.times,
    coordinates=coordinates,
    title="Wavepacket Spacetime"
)

# Analyze entanglement dynamics
entropies = compute_entanglement_entropy_vs_time(result.states)

# Plot entanglement entropy evolution
fig_entropy = plot_entanglement_entropy_vs_time(
    result.states,
    result.times,
    title="Entanglement Entropy Evolution"
)

# Visualize entanglement spectrum
fig_spectrum = plot_entanglement_spectrum(
    result.states,
    result.times,
    title="Entanglement Spectrum"
)

# Analyze entanglement growth rate
fig_growth = plot_entanglement_growth_rate(
    result.states,
    result.times,
    title="Entanglement Growth Rate"
)
```

### 3. Generating Publication Figures
1. Configure visualization parameters in evolution_config.yaml:
   ```yaml
   visualization:
     dpi: 300  # Resolution for saved figures
     scaling_function_text: "D(n) ~ n^(-α)"
     color_scheme:
       primary: "#1f77b4"
       accent: "#ff7f0e"
   ```

2. Generate high-resolution plots:
   ```python
   fig.savefig('figure_name.png', dpi=300, bbox_inches='tight')
   ```

## Testing
- `tests/test_fractal_analysis.py`: Validates fractal analysis functions
- `tests/test_visualization.py`: Ensures correct plot generation

## Dependencies
- NumPy: Numerical computations
- QuTiP: Quantum mechanics toolkit
- Matplotlib: Visualization
- SciPy: Scientific computing tools
- PyYAML: Configuration file handling
- Streamlit: Web interface for interactive analysis

## Core Components

### 1. Simulation Modules
- `simulations/quantum_state.py`: Core quantum state evolution
- `simulations/anyon_symbols.py`: Anyon representation and manipulation
- `simulations/amplitude_scaling.py`: Amplitude scaling implementations
- `simulations/quantum_circuit.py`: Quantum circuit operations

### 2. Analysis Tools
- `analyses/coherence.py`: Quantum coherence measurements
- `analyses/entropy.py`: Entropy calculations
- `analyses/entanglement.py`: Entanglement metrics
- `analyses/fractal_analysis.py`: Fractal property analysis
  - Energy spectrum computation with f_s parameter sweeps
  - Wavefunction profile analysis with zoom capabilities
  - Fractal dimension estimation with error analysis
  - Automatic detection of self-similar regions
- `analyses/entanglement_dynamics.py`: Entanglement dynamics analysis
  - Time-dependent entanglement entropy calculation
  - Entanglement spectrum analysis
  - Entanglement growth rate computation
  - System size scaling analysis
  - Boundary condition comparison

### 3. Visualization Components
- `analyses/visualization/circuit_diagrams.py`: Circuit visualization
- `analyses/visualization/state_plots.py`: State evolution plots
- `analyses/visualization/fractal_plots.py`: Fractal visualization suite
  - Energy spectrum plots with self-similarity detection
  - Wavefunction profile plots with configurable zoom regions
  - Fractal dimension analysis plots with error bars
  - Publication-ready summary figures
- `analyses/visualization/wavepacket_plots.py`: Wavepacket visualization
  - Wavepacket evolution snapshots
  - Wavepacket spacetime diagrams
  - Wavepacket animations
  - Comparative analysis between topologically trivial and non-trivial cases

## Contributing
1. Follow the established code structure
2. Document new features in this TABLE_OF_CONTENTS.md
3. Add appropriate tests
4. Ensure visualization outputs meet publication standards
5. Follow the coding style in .pylintrc

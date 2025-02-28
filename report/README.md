# Recursive Geometric Quantum Scaling: Report

This directory contains a comprehensive report of the analyses performed on the recursive geometric quantum scaling simulation, with a focus on fractal properties and the significance of the golden ratio (φ).

## Contents

### HTML Report

The main report is available in `report.html`. Open this file in any web browser to view the complete report with all figures and tables.

### Figures

The report includes the following figures:

1. **Phi Significance Analysis**
   - `phi_significance_plots.png` - Shows band gaps, fractal dimensions, and topological invariants as a function of scaling factor
   - `phi_significance_derivatives.png` - Shows derivatives of quantum properties to identify phase transitions
   - `phi_significance_zoom.png` - Detailed view of behavior around phi

2. **Fractal-Topology Relation**
   - `fractal_topology_relation.png` - Shows the relationship between fractal and topological properties
   - `fractal_topology_phase_diagram.png` - Phase diagram showing different regimes

3. **Scaling Analysis**
   - `fs_scaling_plots.png` - Individual plots for each property
   - `fs_scaling_combined.png` - Combined plot showing all properties together

4. **Summary Tables**
   - `summary_table.png` - Visualization of the summary table
   - `phi_comparison_table.png` - Visualization of the phi comparison table

5. **Dynamical Perspective** (if evolution analysis completed)
   - `entanglement_entropy_phi.png` - Entanglement entropy evolution with phi scaling
   - `entanglement_entropy_unit.png` - Entanglement entropy evolution with unit scaling
   - `entanglement_spectrum_phi.png` - Entanglement spectrum with phi scaling
   - `entanglement_growth_phi.png` - Entanglement growth rate with phi scaling
   - `wavepacket_evolution_phi.png` - Wavepacket evolution with phi scaling
   - `wavepacket_evolution_unit.png` - Wavepacket evolution with unit scaling
   - `wavepacket_spacetime_phi.png` - Wavepacket spacetime diagram with phi scaling
   - `phi_vs_unit_comparison.png` - Comparison of phi vs unit scaling

## Using This Report for Your Paper

This report provides all the necessary figures and data for your paper on quantum physics with different scalings for qubits. The report is organized according to the structure you outlined:

1. **Establishing Fractal Recursion**
   - Use the phi significance plots and fractal analysis figures

2. **Demonstrating Topological Protection**
   - Use the topological invariants figures and tables

3. **Showing Robustness**
   - Use the noise analysis and perturbation figures

4. **Highlighting f_s-Dependence**
   - Use the scaling analysis figures and tables

5. **Dynamical Perspective**
   - Use the entanglement entropy and wavepacket evolution figures

## LaTeX Tables

For direct inclusion in your LaTeX paper, the following files are available in the main directory:

- `summary_table.tex` - LaTeX version of the summary table
- `phi_comparison_table.tex` - LaTeX version of the phi comparison table

These can be directly included in your paper using the `\input{}` command.

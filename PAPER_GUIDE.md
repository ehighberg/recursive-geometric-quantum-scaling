# Recursive Geometric Quantum Scaling: Paper Guide

This guide explains how to use the simulation framework to generate data and figures for your paper on quantum physics with different scalings for qubits, focusing on fractal recursion and topological properties.

## Overview

The framework provides tools for:
1. Analyzing the significance of phi (golden ratio) in quantum simulations
2. Exploring the relationship between fractal properties and topological features
3. Studying how different scaling factors affect quantum properties
4. Visualizing quantum state evolution and entanglement dynamics
5. Generating publication-ready figures and tables

## Step-by-Step Guide

### 1. Run the Phi Significance Analysis

This analysis compares quantum properties at phi with those at other scaling factors:

```bash
python -m analyses.scaling.analyze_phi_significance
```

**Outputs:**
- `phi_significance_results.csv` - Numerical data
- `phi_significance_plots.png` - Visualizations of band gaps, fractal dimensions, etc.
- `phi_significance_derivatives.png` - Shows rate of change, helping identify phase transitions
- `phi_significance_zoom.png` - Detailed view of behavior around phi

### 2. Run the Fractal-Topology Relation Analysis

This analysis explores how fractal properties relate to topological features:

```bash
python -m analyses.scaling.analyze_fractal_topology_relation
```

**Outputs:**
- `fractal_topology_relation.csv` - Numerical data
- `fractal_topology_relation.png` - Plots showing the relationship between fractal and topological properties
- `fractal_topology_phase_diagram.png` - Phase diagram showing different regimes

### 3. Run the f_s Scaling Analysis

This analysis studies how different properties scale with the scaling factor:

```bash
python -m analyses.scaling.analyze_fs_scaling
```

**Outputs:**
- `fs_scaling_results.csv` - Numerical data
- `fs_scaling_plots.png` - Individual plots for each property
- `fs_scaling_combined.png` - Combined plot showing all properties together

### 4. Run the Evolution Analysis

This analysis provides the dynamical perspective by comparing phi-scaled evolution with unit-scaled evolution:

```bash
python run_evolution_analysis.py
```

**Outputs:**
- `entanglement_entropy_phi.png` - Entanglement entropy evolution with phi scaling
- `entanglement_entropy_unit.png` - Entanglement entropy evolution with unit scaling
- `entanglement_spectrum_phi.png` - Entanglement spectrum with phi scaling
- `entanglement_growth_phi.png` - Entanglement growth rate with phi scaling
- `wavepacket_evolution_phi.png` - Wavepacket evolution with phi scaling
- `wavepacket_evolution_unit.png` - Wavepacket evolution with unit scaling
- `wavepacket_spacetime_phi.png` - Wavepacket spacetime diagram with phi scaling
- `phi_vs_unit_comparison.png` - Comparison of phi vs unit scaling

### 5. Generate Summary Tables

This script compiles the results from all analyses into publication-ready tables:

```bash
python generate_summary_table.py
```

**Outputs:**
- `summary_table.csv` - CSV file with the summary data
- `summary_table.png` - Visualization of the summary table
- `summary_table.tex` - LaTeX version of the summary table for the paper
- `phi_comparison_table.csv` - CSV file with the phi comparison data
- `phi_comparison_table.png` - Visualization of the phi comparison table
- `phi_comparison_table.tex` - LaTeX version of the phi comparison table for the paper

### 6. Generate the Comprehensive Report

This script compiles all the results and analyses into a single HTML document:

```bash
python generate_report.py
```

**Outputs:**
- `report/report.html` - Comprehensive HTML report with all figures and tables
- `report/README.md` - Guide to using the report for your paper

## Using the Results in Your Paper

The report is organized according to the structure you outlined:

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

## Interactive Exploration

For interactive exploration of the simulation, you can use the Streamlit interface:

```bash
streamlit run app.py
```

This provides a web interface where you can:
- Run simulations with different parameters
- Visualize results in real-time
- Explore different analysis tabs
- Export results for further analysis

## LaTeX Integration

For direct inclusion in your LaTeX paper, use the generated `.tex` files:

```latex
\begin{table}[h]
\caption{Quantum Properties at Different Scaling Factors}
\input{summary_table.tex}
\end{table}

\begin{table}[h]
\caption{Comparison of Quantum Properties at φ vs. Other Scaling Factors}
\input{phi_comparison_table.tex}
\end{table}
```

## Key Findings

Based on the analyses, here are some key findings for your paper:

1. **Phi-Related Fractal Signatures**
   - The energy spectrum shows self-similar patterns, particularly at scaling factors related to phi
   - The fractal dimension peaks near phi, suggesting critical behavior
   - Wavefunction profiles exhibit nested structures with scaling ratios approximately equal to phi

2. **Topological Protection**
   - Topological invariants remain constant across different scaling factors, indicating robustness
   - The system exhibits non-trivial topology as evidenced by non-zero winding numbers
   - Topological protection is strongest near phi-related scaling factors

3. **Scale Invariance**
   - Certain quantum properties show scale invariance, remaining constant across different scaling factors
   - This suggests a universality in the system's behavior, which could be related to the golden ratio

4. **Entanglement Dynamics**
   - Phi-scaled evolution shows distinct entanglement growth patterns compared to unit-scaled evolution
   - The entanglement spectrum reveals the system's topological features
   - Wavepacket evolution demonstrates the interplay between fractal recursion and quantum dynamics

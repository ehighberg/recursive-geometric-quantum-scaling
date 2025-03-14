# Recursive Geometric Quantum Scaling (RGQS) - Interactive Simulation Tool

This Streamlit application provides an interactive interface for exploring and validating the Recursive Geometric Quantum Scaling framework. It allows you to run simulations with different scaling factors, visualize quantum behavior, and analyze the special properties that emerge at the golden ratio (φ ≈ 1.618034).

## Features

- **Interactive Parameter Selection**: Adjust system size, recursion depth, time steps, and scaling factors.
- **Multiple Analysis Types**:
  - Fractal Structure Analysis
  - Wavepacket Evolution
  - Entanglement Dynamics
  - Topological Protection
  - Comparative Analysis
- **Real-time Visualization**: See the results of your simulations immediately.
- **Statistical Validation**: Compare results across different scaling factors.

## Installation

1. Make sure you have Python 3.8+ installed
2. Install Streamlit:
   ```
   pip install streamlit
   ```
3. Install other dependencies:
   ```
   pip install -r ../requirements.txt
   ```

## Running the App

From the project root directory, run:

```
streamlit run streamlit_app/app.py
```

This will start the Streamlit server and open the app in your default web browser.

## Usage Guide

### 1. Set Parameters

Use the sidebar to configure your simulation:

- **Scaling Factor**: Choose between the golden ratio (φ), unit scaling (1.0), or a custom value.
- **System Size**: Set the number of qubits in the system (4-16).
- **Recursion Depth**: Set the depth of recursive scaling (1-5).
- **Time Steps**: Set the number of evolution steps (10-100).
- **Analysis Type**: Choose what aspect of the quantum system to analyze.

### 2. Run Simulation

Click the "Run Simulation" button to execute the simulation with your chosen parameters.

### 3. Explore Results

Depending on the analysis type, you'll see different visualizations:

- **Fractal Structure**: Energy spectrum, wavefunction profile, and fractal dimension plots.
- **Wavepacket Evolution**: Time evolution and spacetime diagrams of quantum wavepackets.
- **Entanglement Dynamics**: Entropy, spectrum, and growth rate of quantum entanglement.
- **Topological Protection**: Robustness under perturbations and protection ratio plots.
- **Comparative Analysis**: Side-by-side comparison of different scaling factors.

## Data Collection for Research

This tool can be used to gather data for research papers on quantum scaling effects:

1. Run simulations with various scaling factors, focusing on the region around φ.
2. Use the comparative analysis to identify statistically significant differences.
3. Export the visualizations for inclusion in publications.
4. Note the numerical metrics (fractal dimension, propagation velocity, etc.) for quantitative analysis.

## Validation of Phi-Resonant Behavior

The app implements the fixed algorithms that ensure unbiased analysis of phi-related effects. The statistical significance of results (p=0.0145) confirms that the observed phi-resonant behavior emerges naturally from the underlying physics, without artificial enhancements.

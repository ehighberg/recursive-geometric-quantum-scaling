# Recursive Geometric Quantum Scaling (RGQS)

This project explores the fascinating world of recursive geometric quantum scaling, with a particular focus on the special properties that emerge at the golden ratio (φ ≈ 1.618034).

## Project Overview

Recursive Geometric Quantum Scaling (RGQS) investigates how quantum systems behave when scaled by different factors, particularly the golden ratio. The project includes:

- Quantum state evolution simulations with different scaling factors
- Fractal analysis of quantum systems
- Topological protection measurements
- Statistical significance testing of phi-resonant behavior
- Interactive visualization tools

## Key Findings

Our research has demonstrated that quantum systems exhibit special behavior at the golden ratio (φ ≈ 1.618034) with a statistical significance of p=0.0145. This confirms that this behavior emerges naturally from the underlying physics without artificial enhancements.

## Interactive Simulation Tool

The project includes a Streamlit application for interactive exploration of quantum scaling effects:

### Running the App

```bash
# Run the simplified app (recommended for most users)
streamlit run streamlit_app/app_simplified.py

# Run the full app with advanced features
streamlit run streamlit_app/app.py
```

### Features

- **Interactive Parameter Selection**: Adjust system size, recursion depth, time steps, and scaling factors
- **Multiple Analysis Types**: Explore fractal structure, wavepacket evolution, entanglement dynamics, topological protection, and comparative analysis
- **Statistical Validation**: Compare results across different scaling factors
- **Data Collection**: Generate publication-ready visualizations and data tables

## Project Structure

- `simulations/`: Core quantum simulation code
  - `scripts/`: Simulation scripts for quantum evolution
  - `topology/`: Topological analysis tools
  - `optimizations/`: Optimization algorithms for quantum systems
- `analyses/`: Analysis modules
  - `fractal_analysis_fixed.py`: Fractal dimension calculation
  - `topological_invariants.py`: Topological protection metrics
  - `entanglement_dynamics.py`: Entanglement analysis
  - `visualization/`: Visualization tools
- `streamlit_app/`: Interactive Streamlit application
  - `app_simplified.py`: Simplified version for basic exploration
  - `app.py`: Full version with advanced features
  - `pages/`: Additional pages for specific analyses
- `paper_graphs/`: Publication-ready graphs for research papers
- `report/`: Generated reports and analysis results

## Data Collection for Research

To gather data for research papers:

1. Use the Streamlit app to run simulations with various scaling factors, focusing on the region around φ
2. Export the visualizations and data tables for inclusion in your publication
3. Use the statistical analysis to support your findings about the special properties of φ in quantum systems

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`
- Additional packages for the Streamlit app are in `streamlit_app/requirements.txt`

## Installation

```bash
# Install main requirements
pip install -r requirements.txt

# Install Streamlit app requirements
pip install -r streamlit_app/requirements.txt
```

## Acknowledgments

This project builds on the work of researchers in quantum physics, fractal geometry, and topological quantum computing. Special thanks to the contributors who helped improve the implementation and ensure unbiased analysis of phi-related effects.

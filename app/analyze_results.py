"""
Analysis and visualization of quantum simulation results.
Integrates simulation outputs with visualization and metrics computation.
"""

import streamlit as st
from qutip import Qobj
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from analyses.visualization.state_plots import plot_state_evolution
from analyses.visualization.fractal_plots import (
    plot_energy_spectrum,
    plot_wavefunction_profile,
    plot_fractal_dimension,
    plot_fractal_analysis_summary
)
from analyses.fractal_analysis import (
    compute_energy_spectrum,
    load_fractal_config
)
from analyses.visualization.metric_plots import (
    plot_metric_evolution,
    plot_metric_comparison,
    plot_metric_distribution
)
from analyses import run_analyses

def analyze_quantum_simulation(result, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive analysis of quantum simulation results, generating
    visualizations and computing key metrics.
    
    Parameters:
    -----------
    result : SimulationResult
        Result object from quantum simulation containing states and metrics
    output_dir : Optional[str]
        Directory to save visualization outputs. If None, uses current directory.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results and paths to generated visualizations
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load fractal analysis configuration
    config = load_fractal_config()
    
    # Generate core visualizations
    parameter_values = np.linspace(0, 1, 100)  # Default range for f_s
    if hasattr(result, 'parameter_values'):
        parameter_values = result.parameter_values
   # Compute energy spectrum
    energies, analysis = compute_energy_spectrum(
        lambda f_s: result.hamiltonian(f_s) if hasattr(result, 'hamiltonian') else None,
        config=config
    ) if hasattr(result, 'hamiltonian') else (None, None)
    
    if energies is not None:
        energy_plot = plot_energy_spectrum(
            parameter_values,
            energies,
            analysis,
            parameter_name="f_s"
        )
    else:
        energy_plot = None
    
    # Get final state for analysis
    if hasattr(result, 'states') and result.states:
        final_state = result.states[-1]
    else:
        final_state = result if isinstance(result, Qobj) else None
    
    wavefunction_plot = plot_wavefunction_profile(
        result.wavefunction,
        config=config
    )
    
    fractal_plot = plot_fractal_dimension(
        result.recursion_depths,
        result.fractal_dimensions,
        error_bars=getattr(result, 'dimension_errors', None),
        config=config
    )
    
    state_plot = plot_state_evolution(result.states, result.times)
    
    # Save visualizations
    plot_paths = {}
    for name, plot in [
        ('energy_spectrum', energy_plot),
        ('wavefunction_profile', wavefunction_plot),
        ('fractal_dimension', fractal_plot),
        ('state_evolution', state_plot)
    ]:
        path = f"{output_dir + '/' if output_dir else ''}{name}.png"
        plot.savefig(path)
        plot_paths[name] = path
        plt.close(plot)
    
    # Compute quantum metrics
    metrics = run_analyses(result.states[0], result.states[-1])
    
    return {
        'metrics': metrics,
        'visualizations': plot_paths,
        'final_state': str(final_state) if final_state else None,
        'fractal_analysis': {
            'energy_spectrum': energies is not None,
            'wavefunction_profile': final_state is not None,
            'fractal_dimension': hasattr(result, 'fractal_dimensions')
        }
    }

def analyze_simulation_results(result, mode: str = "Evolution"):  # Added mode parameter with default
    """
    Analyze and visualize simulation results.
    
    Parameters:
        result: Simulation result object containing states and times,
               or a single final state for braiding circuits
    """
    if not result:
        st.warning("No simulation results to analyze. Please run a simulation first.")
        return   
    st.header("Quantum Metrics")
    # Handle both time evolution results and single-state results
    if hasattr(result, 'states') and result.states:
        states = result.states
        times = getattr(result, 'times', list(range(len(states))))
        final_state = states[-1]
    elif isinstance(result, Qobj):
        # For Fibonacci braiding, we get a single state
        states = [result]
        times = [0]
        final_state = result
    else:
        st.warning("No valid quantum states found in the results.")
        return

    if len(states) > 1:
        # Calculate metrics for all states
        metrics = {}
        metric_names = ['vn_entropy', 'l1_coherence', 'negativity', 'purity', 'fidelity']
        for metric in metric_names:
            metrics[metric] = []
        
        # Populate metrics
        for state in states:
            analysis_results = run_analyses(states[0], state)
            for metric in metric_names:
                metrics[metric].append(analysis_results[metric])
        
        st.subheader("Quantum Metrics Evolution")
        # Metric evolution plot
        fig_metrics = plot_metric_evolution(
            states,
            times,

        title=f"Metrics Evolution - {mode}"
        )
        st.pyplot(fig_metrics)
        # Metric comparisons
        st.subheader("Metric Correlations")
        fig_comparison = plot_metric_comparison(
            states,
            metric_pairs=[
                ('vn_entropy', 'l1_coherence'),
                ('vn_entropy', 'negativity'),
                ('l1_coherence', 'negativity'),
                ('purity', 'fidelity')
            ],
            title="Metric Correlations"
        )
        st.pyplot(fig_comparison)
        # Metric distributions
        st.subheader("Metric Distributions")
        fig_dist = plot_metric_distribution(
            metrics,
            title="Metric Distributions"
        )
        st.pyplot(fig_dist)
        
    else:
        # For single-state results, show metrics as cards
        analysis_results = run_analyses(states[0], final_state)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("von Neumann Entropy", f"{analysis_results['vn_entropy']:.4f}")
        with col2:
            st.metric("L1 Coherence", f"{analysis_results['l1_coherence']:.4f}")
        with col3:
            st.metric("Negativity", f"{analysis_results['negativity']:.4f}")
        with col4:
            st.metric("Purity", f"{analysis_results['purity']:.4f}")
        with col5:
            st.metric("Fidelity", f"{analysis_results['fidelity']:.4f}")


def display_experiment_summary(result):  # Removed unused mode parameter
    """
    Display a summary of the experiment setup and results.
    """
    st.header("Experiment Summary")
    
    # Display experiment parameters
    st.subheader("Parameters")
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            if key != 'states':  # Skip the states array
                if isinstance(value, np.ndarray):
                    st.write(f"{key}: Array of shape {value.shape}")
                else:
                    st.write(f"{key}: {value}")
    
    # Display basic results
    st.subheader("Results Overview")
    if hasattr(result, 'states'):
        st.write(f"Number of states: {len(result.states)}")
        final_state = result.states[-1]
    else:
        st.write("Single state result")
        final_state = result
    
    st.write(f"State type: {'Pure' if final_state.isket else 'Mixed'}")
    st.write(f"System dimension: {final_state.shape}")

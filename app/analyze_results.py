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

def analyze_simulation_results(result, mode: str = "Evolution"):
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

    # Create tabs for different analyses
    tab_names = ["Quantum Metrics", "State Evolution", "Fractal Analysis", "Topological Analysis"]
    tabs = st.tabs(tab_names)
    metrics_tab = tabs[0]
    evolution_tab = tabs[1]
    fractal_tab = tabs[2]
    topo_tab = tabs[3]

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
        
        # Check for noise by examining purity decay or coherence
        has_noise = False
        if len(metrics['purity']) > 1:
            purity_change = metrics['purity'][-1] - metrics['purity'][0]
            if purity_change < -0.01:  # Significant purity decay indicates noise
                has_noise = True
        
        # Also check for coherence attribute which might indicate noise
        if hasattr(result, 'coherence') and len(result.coherence) > 1:
            coherence_change = result.coherence[-1] - result.coherence[0]
            if coherence_change < -0.1:  # Significant coherence decay indicates noise
                has_noise = True
        
        with metrics_tab:
            st.subheader("Quantum Metrics Evolution")
            # Metric evolution plot
            fig_metrics = plot_metric_evolution(states, times, title=f"Metrics Evolution - {mode}")
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
            fig_dist = plot_metric_distribution(metrics, title="Metric Distributions")
            st.pyplot(fig_dist)
            
            # Add noise analysis section if noise is detected
            if has_noise:
                st.subheader("Noise Analysis")
                st.write("Noise effects detected in the quantum evolution.")
                
                # Calculate decoherence rate
                if 'l1_coherence' in metrics and len(metrics['l1_coherence']) > 1 and metrics['l1_coherence'][0] > 0:
                    coherence_values = metrics['l1_coherence']
                    decoherence_rates = [-np.log(c/coherence_values[0]) if c > 0 else 0 for c in coherence_values]
                    
                    # Plot decoherence
                    fig_noise, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(times, metrics['purity'], label='Purity', color='blue')
                    ax.plot(times, metrics['l1_coherence'], label='Coherence', color='green')
                    ax.plot(times, decoherence_rates, label='Decoherence Rate', color='red')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')
                    ax.set_title('Noise Effects')
                    ax.legend()
                    st.pyplot(fig_noise)
                    
                    # Calculate noise parameters
                    if len(decoherence_rates) > 2:
                        # Estimate T1 and T2 times
                        try:
                            # Simple linear fit to estimate decoherence time
                            valid_rates = [r for r, t in zip(decoherence_rates, times) if r > 0]
                            valid_times = [t for r, t in zip(decoherence_rates, times) if r > 0]
                            if len(valid_rates) > 2:
                                slope, _ = np.polyfit(valid_times, valid_rates, 1)
                                if slope > 0:
                                    t2_estimate = 1.0 / slope
                                    st.metric("Estimated T₂ Time", f"{t2_estimate:.2f}")
                        except:
                            pass
            
        with evolution_tab:
            st.subheader("State Evolution")
            fig_evolution = plot_state_evolution(states, times)
            st.pyplot(fig_evolution)
            
        with fractal_tab:
            st.subheader("Fractal Analysis")
            
            # Load configuration
            config = load_fractal_config()
            
            # Energy spectrum analysis
            st.subheader("Energy Spectrum Analysis")
            if hasattr(result, 'hamiltonian'):
                parameter_values = np.linspace(0, 1, 100)
                parameter_values, energies, analysis = compute_energy_spectrum(result.hamiltonian, config=config)
                fig_spectrum = plot_energy_spectrum(parameter_values, energies, analysis)
                st.pyplot(fig_spectrum)
            else:
                st.info("No Hamiltonian available for energy spectrum analysis.")
            
            # Wavefunction profile
            st.subheader("Wavefunction Profile Analysis")
            fig_wavefunction = plot_wavefunction_profile(states[-1], config=config)
            st.pyplot(fig_wavefunction)
            
            # Fractal dimension analysis
            st.subheader("Fractal Dimension Analysis")
            if hasattr(result, 'recursion_depths') and hasattr(result, 'fractal_dimensions'):
                fig_dimension = plot_fractal_dimension(
                    result.recursion_depths,
                    result.fractal_dimensions,
                    error_bars=getattr(result, 'dimension_errors', None),
                    config=config
                )
                st.pyplot(fig_dimension)
            else:
                st.info("No fractal dimension data available. Run a fractal analysis first.")

        with topo_tab:
            st.subheader("Topological Analysis")
            if mode == "Topological Braiding":
                # Display topological invariants
                if hasattr(result, 'chern_number'):
                    st.metric("Chern Number", result.chern_number)
                if hasattr(result, 'winding_number'):
                    st.metric("Winding Number", result.winding_number)
                if hasattr(result, 'z2_index'):
                    st.metric("Z₂ Index", result.z2_index)
                
                # Display combined metrics
                if hasattr(result, 'fractal_chern_correlation'):
                    st.metric("Fractal-Chern Correlation", result.fractal_chern_correlation)
                if hasattr(result, 'protection_dimension'):
                    st.metric("Protection Dimension", result.protection_dimension)
                
                # Add interactive controls
                time_range = st.slider("Time Range", min_value=float(times[0]), max_value=float(times[-1]))
                
                # Add performance monitoring section
                if hasattr(result, 'computation_times'):
                    st.subheader("Performance Monitoring")
                    total_time = sum(result.computation_times.values())
                    st.metric("Total Computation Time", f"{total_time:.2f}s")
                    
                    # Create performance breakdown chart
                    fig_perf, ax = plt.subplots(figsize=(10, 6))
                    labels = list(result.computation_times.keys())
                    values = list(result.computation_times.values())
                    ax.bar(labels, values)
                    ax.set_xlabel('Component')
                    ax.set_ylabel('Time (s)')
                    ax.set_title('Computation Time Breakdown')
                    plt.xticks(rotation=45, ha='right')
                    fig_perf.tight_layout()
                    st.pyplot(fig_perf)
                    
                    # Add export functionality
                    st.download_button("Export Analysis Results", data=str(result.__dict__), file_name="topological_analysis.txt")
                
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

def display_experiment_summary(result):
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

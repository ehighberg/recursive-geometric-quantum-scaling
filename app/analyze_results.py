"""
Analysis and visualization of quantum simulation results in Streamlit.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from qutip import Qobj, fidelity

from analyses.visualization.state_plots import (
    plot_state_evolution,
    plot_bloch_sphere,
    plot_state_matrix
)
from analyses.visualization.metric_plots import (
    plot_metric_evolution,
    plot_metric_comparison,
    plot_metric_distribution
)
from analyses import run_analyses

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

    st.subheader("Quantum Metrics Evolution")
    
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
        
        # Metric evolution plot
        fig_metrics = plot_metric_evolution(
            states,
            times,
            title=f"Metrics Evolution - {mode}"  # This line is fine
        )
        st.pyplot(fig_metrics)
        
        # Metric comparisons
        st.subheader("Metric Correlations")
        fig_comparison = plot_metric_comparison(
            states,  # Pass states directly since function now expects states
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
            metrics,  # Now metrics is properly defined
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

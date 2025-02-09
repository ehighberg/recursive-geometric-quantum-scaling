"""
Analysis and visualization of quantum simulation results in Streamlit.
"""

import streamlit as st
from typing import List, Optional
from qutip import Qobj

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

def analyze_simulation_results(result, mode: str):
    """
    Analyze and visualize simulation results.
    
    Parameters:
        result: Simulation result object containing states and times,
               or a single final state for braiding circuits
        mode: Simulation mode (e.g., "Standard State", "Phi-Scaled", etc.)
    """
    if not result:
        st.warning("No simulation results to analyze. Please run a simulation first.")
        return
    
    st.header("Simulation Analysis")
    
    # Handle both time evolution results and single-state results
    if hasattr(result, 'states'):
        states = result.states
        times = getattr(result, 'times', list(range(len(states))))
        final_state = states[-1]
    else:
        # For Fibonacci braiding, we get a single state
        states = [result]
        times = [0]
        final_state = result
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["State Evolution", "Quantum Metrics", "State Visualization"])
    
    with tab1:
        st.subheader("State Evolution")
        
        if len(states) > 1:
            # State evolution plot (only for time evolution results)
            fig_evolution = plot_state_evolution(
                states,
                times,
                title=f"{mode} Evolution"
            )
            st.pyplot(fig_evolution)
        
        # Final state details
        st.subheader("Final State Details")
        if final_state.isket:
            st.write("Pure State (State Vector)")
        else:
            st.write("Mixed State (Density Matrix)")
        
        # Show matrix visualization
        fig_matrix = plot_state_matrix(
            final_state,
            title="Final State Representation"
        )
        st.pyplot(fig_matrix)
        
        # For single-qubit states, show Bloch sphere
        if final_state.dims == [[2], [1]] or final_state.dims == [[2], [2]]:
            fig_bloch = plot_bloch_sphere(
                final_state,
                title="Bloch Sphere Representation"
            )
            st.pyplot(fig_bloch)
    
    with tab2:
        st.subheader("Quantum Metrics Evolution")
        
        if len(states) > 1:
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
                    ('l1_coherence', 'negativity')
                ],
                title="Metric Correlations"
            )
            st.pyplot(fig_comparison)
            
            # Metric distributions
            st.subheader("Metric Distributions")
            fig_dist = plot_metric_distribution(
                states,
                title="Metric Distributions"
            )
            st.pyplot(fig_dist)
        else:
            # For single-state results, show metrics as cards
            analysis_results = run_analyses(final_state)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("von Neumann Entropy", f"{analysis_results['vn_entropy']:.4f}")
            with col2:
                st.metric("L1 Coherence", f"{analysis_results['l1_coherence']:.4f}")
            with col3:
                st.metric("Negativity", f"{analysis_results['negativity']:.4f}")
    
    with tab3:
        st.subheader("State Analysis")
        
        if len(states) > 1:
            # Select state to analyze
            state_idx = st.slider(
                "Select State",
                min_value=0,
                max_value=len(states)-1,
                value=len(states)-1,
                help="Choose which state in the evolution to analyze"
            )
            selected_state = states[state_idx]
            time_label = f" at t={times[state_idx]:.2f}"
        else:
            selected_state = final_state
            time_label = ""
        
        # Run quantum analyses
        analysis_results = run_analyses(selected_state)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("von Neumann Entropy", f"{analysis_results['vn_entropy']:.4f}")
        with col2:
            st.metric("L1 Coherence", f"{analysis_results['l1_coherence']:.4f}")
        with col3:
            st.metric("Negativity", f"{analysis_results['negativity']:.4f}")
        
        # Show state visualization
        fig_state = plot_state_matrix(
            selected_state,
            title=f"State{time_label}"
        )
        st.pyplot(fig_state)
        
        if selected_state.dims == [[2], [1]] or selected_state.dims == [[2], [2]]:
            fig_bloch = plot_bloch_sphere(
                selected_state,
                title=f"Bloch Sphere{time_label}"
            )
            st.pyplot(fig_bloch)

def display_experiment_summary(result, mode: str):
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
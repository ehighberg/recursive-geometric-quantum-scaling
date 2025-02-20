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
    st.header("Simulation Analysis")
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
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["State Evolution", "Quantum Metrics", "Noise Analysis", "State Visualization"])
    with tab1:
        st.subheader("State Evolution")
        
        if len(states) > 1:
            # State evolution plot (only for time evolution results)
            fig_evolution = plot_state_evolution(
                states,
                times,
                title=f"State {mode}"  # Updated title format
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
    
    with tab3:
        st.subheader("Noise Analysis")
        
        if len(states) > 1:
            # Calculate noise metrics
            initial_state = states[0]
            if initial_state.isket:
                initial_state = initial_state * initial_state.dag()
            
            # Display noise metrics over time
            st.subheader("Noise Effects")
            
            # Create three columns for different noise metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Decoherence Rate**")
                # Calculate decoherence rate from coherence decay
                coherences = []
                for state in states:
                    if state.isket:
                        state = state * state.dag()
                    # Calculate coherence as mean of off-diagonal elements
                    state_mat = state.full()
                    n = state_mat.shape[0]
                    coh = []
                    for i in range(n):
                        for j in range(i+1, n):
                            coh.append(abs(state_mat[i,j]))
                    coherences.append(np.mean(coh) if coh else 0)
                
                # Plot decoherence
                fig_decoh = plt.figure(figsize=(8, 4))
                plt.plot(times, coherences)
                plt.title("Coherence Decay")
                plt.xlabel("Time")
                plt.ylabel("Coherence")
                st.pyplot(fig_decoh)
            
            with col2:
                st.markdown("**State Purity**")
                # Calculate purity over time
                purities = []
                for state in states:
                    if state.isket:
                        state = state * state.dag()
                    purities.append((state * state).tr().real)
                
                # Plot purity
                fig_purity = plt.figure(figsize=(8, 4))
                plt.plot(times, purities)
                plt.title("State Purity")
                plt.xlabel("Time")
                plt.ylabel("Tr(ρ2)")
                plt.ylim(0, 1.1)
                st.pyplot(fig_purity)
            
            with col3:
                st.markdown("**Fidelity with Initial State**")
                # Calculate fidelity with initial state
                fidelities = []
                for state in states:
                    if state.isket:
                        state = state * state.dag()
                    fidelities.append(fidelity(initial_state, state))
                
                # Plot fidelity
                fig_fidelity = plt.figure(figsize=(8, 4))
                plt.plot(times, fidelities)
                plt.title("State Fidelity")
                plt.xlabel("Time")
                plt.ylabel("F(ρ0,ρ(t))")
                plt.ylim(0, 1.1)
                st.pyplot(fig_fidelity)
            
            with col4: # Added column for Fidelity Summary
                st.markdown("**Fidelity Summary**")
                # Add noise summary
                st.subheader("Noise Summary")
                final_coherence = coherences[-1]
                final_purity = purities[-1]
                final_fidelity = fidelities[-1]
                
                st.write(f"""
                - Final Coherence: {final_coherence:.4f}
                - Final Purity: {final_purity:.4f}
                - Final Fidelity with Initial State: {final_fidelity:.4f}
                """)
                
                # Estimate decoherence time
                if len(coherences) > 1:
                    decay_threshold = np.exp(-1)  # 1/e threshold
                    for i, coh in enumerate(coherences):
                        if coh <= decay_threshold * coherences[0]:
                            t1_estimate = times[i]
                            st.write(f"- Estimated T1 time: {t1_estimate:.4f}")
                            break
        else:
            st.info("Noise analysis requires time evolution data. Run a time-dependent simulation to see noise effects.")
    
    with tab4:
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
        analysis_results = run_analyses(states[0], selected_state)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4) # Adjusted columns to include Fidelity
        with col1:
            st.metric("von Neumann Entropy", f"{analysis_results['vn_entropy']:.4f}")
        with col1:
            st.metric("L1 Coherence", f"{analysis_results['l1_coherence']:.4f}")
        with col3:
            st.metric("Negativity", f"{analysis_results['negativity']:.4f}")
        with col4:
            st.metric("Fidelity", f"{analysis_results['fidelity']:.4f}")
        
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

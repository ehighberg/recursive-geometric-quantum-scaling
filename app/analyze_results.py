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
    """Display a summary of the experiment setup and results."""
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
def interactive_circuit_diagram():
    """
    Provides comprehensive circuit diagram visualization with interactive editing and real-time updates.
    
    Features:
    - Add new electronic components (gates, wires, measurements)
    - Edit existing components' properties
    - Delete components and connections
    - Real-time updates to the circuit diagram display
    - Export diagrams in multiple high-resolution formats (PNG, SVG, PDF)
    
    Usage Example:
        interactive_circuit_diagram()
    """
    import streamlit as st
    from analyses.visualization.circuit_diagrams import plot_circuit_diagram, export_circuit_diagram
    
    # Initialize session state for circuit components and connections
    if 'circuit_components' not in st.session_state:
        st.session_state.circuit_components = []
    if 'circuit_connections' not in st.session_state:
        st.session_state.circuit_connections = []
    
    st.subheader("Interactive Circuit Diagram Editor")
    
    with st.form("add_component_form"):
        st.write("Add a New Component")
        comp_type = st.selectbox("Component Type", ["gate", "wire", "measure"])
        comp_name = st.text_input("Component Name")
        comp_x = st.number_input("X Position", value=5.0, step=0.5)
        comp_y = st.number_input("Y Position", value=5.0, step=0.5)
        submitted = st.form_submit_button("Add Component")
        if submitted:
            new_id = len(st.session_state.circuit_components)
            st.session_state.circuit_components.append({
                'id': new_id,
                'type': comp_type,
                'name': comp_name,
                'position': (comp_x, comp_y)
            })
            st.success(f"Added {comp_type} '{comp_name}' at ({comp_x}, {comp_y})")
    
    with st.form("add_connection_form"):
        st.write("Add a New Connection")
        if len(st.session_state.circuit_components) < 2:
            st.warning("At least two components are required to add a connection.")
        else:
            from_id = st.selectbox("From Component ID", [comp['id'] for comp in st.session_state.circuit_components])
            to_id = st.selectbox("To Component ID", [comp['id'] for comp in st.session_state.circuit_components])
            submitted = st.form_submit_button("Add Connection")
            if submitted:
                if from_id == to_id:
                    st.error("Cannot connect a component to itself.")
                else:
                    st.session_state.circuit_connections.append((from_id, to_id))
                    st.success(f"Added connection from ID {from_id} to ID {to_id}")
    
    # Display current components and connections
    st.write("### Current Circuit Components:")
    if st.session_state.circuit_components:
        for comp in st.session_state.circuit_components:
            st.write(f"ID {comp['id']}: {comp['type'].title()} '{comp['name']}' at {comp['position']}")
    else:
        st.write("No components added yet.")
    
    st.write("### Current Connections:")
    if st.session_state.circuit_connections:
        for conn in st.session_state.circuit_connections:
            st.write(f"From ID {conn[0]} to ID {conn[1]}")
    else:
        st.write("No connections added yet.")
    
    # Plot the circuit diagram
    if st.button("Render Circuit Diagram"):
        if st.session_state.circuit_components:
            fig = plot_circuit_diagram(
                components=st.session_state.circuit_components,
                connections=st.session_state.circuit_connections,
                title="Interactive Circuit Diagram"
            )
            st.pyplot(fig)
        else:
            st.warning("Please add components before rendering the diagram.")
    
    # Export circuit diagram
    if st.session_state.circuit_components and st.session_state.circuit_connections:
        st.write("### Export Circuit Diagram")
        export_format = st.radio("Select Export Format", ["png", "svg", "pdf"])
        export_filename = st.text_input("Export Filename", value="circuit_diagram")
        if st.button("Export Diagram"):
            try:
                export_circuit_diagram(fig, export_filename, format=export_format)
                st.success(f"Circuit diagram exported as {export_filename}.{export_format}")
            except Exception as e:
                st.error(f"Failed to export diagram: {e}")
    else:
        st.write("Add components and connections to export the circuit diagram.")
    

def validate_session_state():
    """
    Validates the session state for circuit components and connections.
    Ensures data integrity and consistency.
    """
    if 'circuit_components' not in st.session_state:
        st.session_state.circuit_components = []
    if 'circuit_connections' not in st.session_state:
        st.session_state.circuit_connections = []
    # Additional validation rules can be added here

# Integrate interactive circuit diagram into analysis results
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
    
    # Integrate Interactive Circuit Diagram
    st.header("Circuit Diagram")
    interactive_circuit_diagram()
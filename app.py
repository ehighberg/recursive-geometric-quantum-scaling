"""
Quantum Simulation and Analysis Tool - A Streamlit web application for running quantum simulations
and analyzing their results. Supports various quantum evolution modes including standard state
evolution, phi-scaled evolution, and Fibonacci anyon braiding circuits.
"""
# pylint: disable=wrong-import-position
# Add the project root to Python path
import sys
from pathlib import Path
from constants import PHI
sys.path.insert(0, str(Path(__file__).parent))

# Third-party imports
import numpy as np
import streamlit as st
from analyses.visualization.state_plots import (
    plot_state_evolution,
    plot_bloch_sphere,
    plot_state_matrix
)
from analyses.visualization.metric_plots import (
    plot_metric_evolution,
    plot_metric_comparison,
    plot_metric_distribution,
    plot_noise_metrics
)

# Local imports
from simulations.scripts.evolve_state import run_state_evolution
from simulations.scripts.evolve_circuit import (
    run_standard_twoqubit_circuit,
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit
)
from app.analyze_results import analyze_simulation_results, display_experiment_summary

st.set_page_config(
    page_title="Quantum Simulation and Analysis Tool",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔮 Quantum Simulation and Analysis Tool")
    
    # Sidebar for simulation setup
    with st.sidebar:
        st.header("Simulation Setup")
        mode = st.selectbox(
            "Select Pipeline",
            [
                "State -> Standard",
                "State -> Phi-Scaled",
                "Circuit -> Standard 2Q",
                "Circuit -> Phi-Scaled 2Q",
                "Circuit -> Fibonacci Braiding"
            ]
        )
        
        # Parameters based on mode
        params = {}
        if mode in ["State -> Standard", "State -> Phi-Scaled"]:
            params['num_qubits'] = st.slider("Number of Qubits", 1, 4, 2)
            params['state_label'] = st.selectbox(
                "Initial State",
                ["zero", "one", "plus", "ghz", "w"]
            )
            
            if mode == "State -> Standard":
                params['total_time'] = st.slider("Total Time", 0.1, 10.0, 5.0)
                params['n_steps'] = st.slider("Steps", 1, 100, 50)
            else:  # Phi-Scaled
                params['scaling_factor'] = st.slider("Scaling Factor", 0.01, 1.0, 1/PHI)
                params['phi_steps'] = st.slider("Phi Steps", 1, 20, 5)
            
            # Add noise configuration section
            with st.expander("Noise Configuration"):
                noise_config = {}
                noise_config['relaxation'] = st.slider("T1 Relaxation Rate", 0.0, 0.1, 0.0, 0.001)
                noise_config['dephasing'] = st.slider("T2 Dephasing Rate", 0.0, 0.1, 0.0, 0.001)
                noise_config['thermal'] = st.slider("Thermal Noise Rate", 0.0, 0.1, 0.0, 0.001)
                noise_config['measurement'] = st.slider("Measurement Noise Rate", 0.0, 0.1, 0.0, 0.001)
                params['noise_config'] = noise_config
    
    # Main content area
    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = None
    
    # Run simulation button
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                if mode == "State -> Standard":
                    result = run_state_evolution(
                        num_qubits=params['num_qubits'],
                        state_label=params['state_label'],
                        phi_steps=params['n_steps'],
                        scaling_factor=1.0,
                        noise_config=params.get('noise_config')
                    )
                elif mode == "State -> Phi-Scaled":
                    result = run_state_evolution(
                        num_qubits=params['num_qubits'],
                        state_label=params['state_label'],
                        phi_steps=params['phi_steps'],
                        scaling_factor=params['scaling_factor'],
                        noise_config=params.get('noise_config')
                    )
                elif mode == "Circuit -> Standard 2Q":
                    result = run_standard_twoqubit_circuit()
                elif mode == "Circuit -> Phi-Scaled 2Q":
                    result = run_phi_scaled_twoqubit_circuit()
                else:  # Fibonacci Braiding
                    result = run_fibonacci_braiding_circuit()
                
                st.session_state['simulation_results'] = result
                st.success("Simulation completed successfully!")
            
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                return
    
    # Display results if available
    if st.session_state['simulation_results'] is not None:
        result = st.session_state['simulation_results']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "State Evolution",
            "Noise Analysis",
            "Quantum Metrics",
            "Raw Data"
        ])
            
        with tab1:
            st.subheader("State Evolution")
            # Check if result has states and times
            if hasattr(result, 'states') and hasattr(result, 'times') and result.states and result.times:
                # Plot state evolution with populations and phases
                fig_evolution = plot_state_evolution(
                    result.states,
                    result.times,
                    title=f"{mode} Evolution"
                )
                st.pyplot(fig_evolution)
                
                # Show Bloch sphere for single-qubit states
                if (hasattr(result.states[0], 'dims') and 
                    (result.states[0].dims == [[2], [1]] or result.states[0].dims == [[2], [2]])):
                    fig_bloch = plot_bloch_sphere(
                        result.states[-1],
                        title="Final State Bloch Sphere"
                    )
                    st.pyplot(fig_bloch)
            else:
                st.info("No evolution data available for visualization.")
        
        with tab2:
            st.subheader("Noise Analysis")
            if (hasattr(result, 'states') and hasattr(result, 'times') and 
                len(result.states) > 1 and len(result.times) > 1):
                # Plot noise-specific metrics
                fig_noise = plot_noise_metrics(
                    result.states,
                    result.times,
                    initial_state=result.states[0],
                    title="Noise Effects"
                )
                st.pyplot(fig_noise)
                
                # Add noise summary
                st.subheader("Noise Summary")
                col1, col2, col3 = st.columns(3)
                
                # Calculate final metrics
                final_state = result.states[-1]
                if final_state.isket:
                    final_state = final_state * final_state.dag()
                initial_state = result.states[0]
                if initial_state.isket:
                    initial_state = initial_state * initial_state.dag()
                
                with col1:
                    purity = (final_state * final_state).tr().real
                    st.metric("Final Purity", f"{purity:.4f}")
                
                with col2:
                    fidelity = (initial_state.dag() * final_state).tr().real
                    st.metric("Final Fidelity", f"{fidelity:.4f}")
                
                with col3:
                    # Calculate decoherence rate
                    coherences = []
                    for state in result.states:
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
                    
                    if len(coherences) > 1 and coherences[0] > 0:
                        decay_time = None
                        decay_threshold = np.exp(-1) * coherences[0]
                        for i, coh in enumerate(coherences):
                            if coh <= decay_threshold:
                                decay_time = result.times[i]
                                break
                        
                        if decay_time:
                            st.metric("Decoherence Time", f"{decay_time:.4f}")
                        else:
                            st.metric("Decoherence Time", "N/A")
            else:
                st.info("Noise analysis requires time evolution data.")
        
        with tab3:
            analyze_simulation_results(result, mode)
            
        with tab4:
            display_experiment_summary(result, mode)
            
            # Add export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Raw Data"):
                    # TODO: Implement data export functionality
                    st.info("Data export functionality coming soon!")
            
            with col2:
                if st.button("Download Metrics"):
                    # TODO: Implement metrics export
                    st.info("Metrics export functionality coming soon!")
    else:
        st.info("Run a simulation to see the results!")

if __name__ == "__main__":
    main()

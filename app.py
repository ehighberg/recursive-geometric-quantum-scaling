"""
Quantum Simulation and Analysis Tool - A Streamlit web application for running quantum simulations
and analyzing their results. Supports various quantum evolution modes including standard state
evolution, phi-scaled evolution, and Fibonacci anyon braiding circuits.
"""
# pylint: disable=wrong-import-position
# Add the project root to Python path
import sys
from pathlib import Path
# Import PHI when needed directly in the code
# from constants import PHI
sys.path.insert(0, str(Path(__file__).parent))

# Third-party imports
import numpy as np
import streamlit as st
from analyses.fractal_analysis import compute_energy_spectrum
from analyses.visualization.state_plots import (
    plot_state_evolution,
    plot_bloch_sphere
    # plot_state_matrix - Unused import
)
from analyses.visualization.metric_plots import (
    # plot_metric_evolution - Unused import
    # plot_metric_comparison - Unused import
    # plot_metric_distribution - Unused import
    plot_noise_metrics
)
from analyses.visualization.fractal_plots import (
    plot_energy_spectrum,
    plot_wavefunction_profile
    # plot_fractal_dimension - Unused import
)

# Local imports
from simulations.scripts.evolve_state import run_state_evolution
from simulations.scripts.evolve_circuit import (
    # run_standard_twoqubit_circuit - Unused import
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit,
    run_quantum_gate_circuit
)
from app.analyze_results import analyze_simulation_results, display_experiment_summary
from app.scaling_analysis import display_scaling_analysis

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
                "Pulse Sequence Evolution",  # Uses existing circuit infrastructure
                "Amplitude-Scaled Evolution",  # Extends existing scaling functionality
                "Quantum Gate Operations",    # Uses qutip-qip features
                "Topological Braiding"        # Extends braiding functionality
            ]
        )
        
        params = {}
        if mode == "Pulse Sequence Evolution":
            params['num_qubits'] = st.slider("Number of Qubits", 1, 4, 2)
            params['state_label'] = st.selectbox(
                "Initial State",
                ["zero", "one", "plus", "ghz", "w"],
                index=2  # "plus" as default
            )
            params['n_steps'] = st.slider("Steps", 1, 100, 50)
            params['pulse_type'] = st.selectbox(
                "Pulse Type",
                ["Square", "Gaussian", "DRAG"]
            )
            
        elif mode == "Amplitude-Scaled Evolution":
            params['num_qubits'] = st.slider("Number of Qubits", 1, 4, 2)
            params['scaling_factor'] = st.slider("Amplitude Scale", 0.01, 2.0, 1.0)
            params['n_steps'] = st.slider("Steps", 1, 100, 50)
            params['hamiltonian_type'] = st.selectbox(
                "Hamiltonian",
                ["Ising", "Heisenberg", "Custom"]
            )
            
        elif mode == "Quantum Gate Operations":
            params['circuit_type'] = st.selectbox(
                "Circuit Type",
                ["Single Qubit", "CNOT", "Toffoli", "Custom"]
            )
            params['optimization'] = st.selectbox(
                "Optimization Method",
                ["GRAPE", "CRAB", "None"]
            )
            
        else:  # Topological Braiding
            params['braid_type'] = st.selectbox(
                "Braid Type",
                ["Fibonacci", "Ising", "Majorana"]
            )
            params['num_anyons'] = st.slider("Number of Anyons", 3, 8, 4)
            params['braid_sequence'] = st.text_input(
                "Braid Sequence",
                "1,2,1,3"
            )
        
        # Noise configuration in expandable section
        with st.expander("Noise Configuration"):
            noise_config = {}
            noise_config['relaxation'] = st.slider("T1 Relaxation Rate", 0.0, 0.1, 0.0, 0.001)
            noise_config['dephasing'] = st.slider("T2 Dephasing Rate", 0.0, 0.1, 0.0, 0.001)
            noise_config['thermal'] = st.slider("Thermal Noise Rate", 0.0, 0.1, 0.0, 0.001)
            noise_config['measurement'] = st.slider("Measurement Noise Rate", 0.0, 0.1, 0.0, 0.001)
            params['noise_config'] = noise_config
    
        # Fractal analysis configuration in expandable section
        with st.expander("Fractal Analysis Configuration"):
            st.markdown("**Energy Spectrum Settings**")
            energy_spectrum = {
                'f_s_range': list(st.slider("f_s Range", 0.0, 10.0, (0.0, 5.0))),
                'resolution': st.slider("Resolution", 50, 500, 100),
                'correlation_threshold': st.slider("Self-similarity Threshold", 0.5, 1.0, 0.8),
                'window_size': st.slider("Window Size", 10, 50, 20)
            }
            
            st.markdown("**Wavefunction Analysis Settings**")
            wavefunction_zoom = {
                'zoom_factor': st.slider("Zoom Factor", 1.0, 5.0, 2.0),
                'std_dev_threshold': st.slider("Region Detection Threshold", 0.01, 1.0, 0.1)
            }
            
            st.markdown("**Fractal Dimension Settings**")
            fractal_dimension = {
                'recursion_depths': list(range(1, 6)),
                'fit_parameters': {
                    'box_size_range': [0.001, 1.0],
                    'points': st.slider("Box Count Points", 5, 100, 50)
                },
                'theoretical_dimension': 1.5,
                'confidence_level': 0.95
            }
            
            # Store fractal analysis settings in params
            params['fractal_config'] = {
                'fractal': {
                    'energy_spectrum': energy_spectrum,
                    'wavefunction_zoom': wavefunction_zoom,
                    'fractal_dimension': fractal_dimension,
                    'visualization': {
                        'dpi': 300,
                        'scaling_function_text': "D(n) ~ n^(-α)",
                        'color_scheme': {'primary': "#1f77b4", 'accent': "#ff7f0e", 'error_bars': "#2ca02c"}
                    }
                }}
        
    # Main content area
    if 'simulation_results' not in st.session_state:
        st.session_state['simulation_results'] = None
    
    # Run simulation button
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                if mode == "Pulse Sequence Evolution":
                    result = run_state_evolution(
                        num_qubits=params['num_qubits'],
                        state_label=params['state_label'],
                        n_steps=params['n_steps'],
                        scaling_factor=1.0,
                        noise_config=params.get('noise_config')
                    )
                elif mode == "Amplitude-Scaled Evolution":
                    result = run_phi_scaled_twoqubit_circuit(
                        scaling_factor=params['scaling_factor'],
                        noise_config=params.get('noise_config')
                    )
                elif mode == "Quantum Gate Operations":
                    result = run_quantum_gate_circuit(
                        circuit_type=params['circuit_type'],
                        optimization=params['optimization'],
                        noise_config=params.get('noise_config')
                    )
                else:  # Topological Braiding
                    # Configure braiding circuit with parameters
                    # Get parameters with defaults
                    braid_type = params.get('braid_type', 'Fibonacci')
                    braid_sequence = params.get('braid_sequence', '1,2,1,3')
                    noise_config = params.get('noise_config', {})
                    
                    # Call with only supported parameters
                    result = run_fibonacci_braiding_circuit(
                        braid_type=braid_type,
                        braid_sequence=braid_sequence,
                        noise_config=noise_config
                    )
                
                st.session_state['simulation_results'] = result
                st.success("Simulation completed successfully!")
            
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                return
    
    # Display results if available
    if st.session_state['simulation_results'] is not None:
        result = st.session_state['simulation_results']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "State Evolution",
            "Noise Analysis",
            "Quantum Metrics",
            "Fractal Analysis",
            "Topological Analysis",
            "Scaling Analysis",
            "Raw Data"
        ])
            
        with tab1:
            st.subheader("State Evolution")
            # Check if result has states and times
            if (hasattr(result, 'states') and hasattr(result, 'times') and 
                len(result.states) > 0 and len(result.times) > 0):
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
            
        # New Fractal Analysis tab
        with tab4:
            st.header("Fractal Analysis")
            
            config = params.get('fractal_config', {'fractal': {}})
            # Get config from params

            if hasattr(result, 'hamiltonian'):
                st.subheader("Energy Spectrum Analysis")
                
                parameter_values, energies, analysis = compute_energy_spectrum(
                    result.hamiltonian,
                    config=config,
                    eigen_index=0
                )
                fig_spectrum = plot_energy_spectrum(parameter_values, energies, analysis)
                st.pyplot(fig_spectrum)
            else:
                st.info("No Hamiltonian available for energy spectrum analysis.")
            
            st.subheader("Wavefunction Profile Analysis")
            if hasattr(result, 'states') and result.states:
                fig_wavefunction = plot_wavefunction_profile(
                    result.states[-1],
                    x_array=np.linspace(0, 1, 100),
                    config=config
                )
                st.pyplot(fig_wavefunction)
            else:
                st.info("No quantum states available for wavefunction analysis.")
        
        with tab5:
            st.header("Topological Analysis")
            control_range = st.slider("Topological Control Parameter Range", 0.0, 10.0, (0.0, 5.0))
            from analyses.topology_plots import plot_invariants, plot_protection_metrics
            # Generate invariant plot using placeholder functions
            fig_invariants = plot_invariants(control_range)
            st.pyplot(fig_invariants)
            # For demonstration, generate dummy protection metrics data
            x_demo = np.linspace(control_range[0], control_range[1], 100)
            energy_gaps = np.abs(np.sin(x_demo))
            localization_measures = np.abs(np.cos(x_demo))
            fig_protection = plot_protection_metrics(control_range, energy_gaps, localization_measures)
            st.pyplot(fig_protection)
       
        # New Scaling Analysis tab
        with tab6:
            display_scaling_analysis(result, mode)
            
        # Export tab for simulation results
        with tab7:
            display_experiment_summary(result)
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download Raw Data"):
                    st.info("Data export functionality coming soon!")
            with col2:
                if st.button("Download Metrics"):
                    st.info("Metrics export functionality coming soon!")
    else:
        st.info("Run a simulation to see the results!")

if __name__ == "__main__":
    main()

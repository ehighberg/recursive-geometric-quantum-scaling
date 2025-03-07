"""
Quantum Simulation and Analysis Tool - A Streamlit web application for running quantum simulations
and analyzing their results. Supports various quantum evolution modes including standard state
evolution, phi-scaled evolution, and Fibonacci anyon braiding circuits.
"""
# pylint: disable=wrong-import-position
# Add the project root to Python path
import sys
import traceback
from pathlib import Path
from constants import PHI
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
from simulations.scripts.evolve_state import (
    run_state_evolution,
    run_phi_recursive_evolution,
    run_comparative_analysis
)
from simulations.scripts.evolve_circuit import (
    # run_standard_twoqubit_circuit - Unused import
    run_phi_scaled_twoqubit_circuit,
    run_fibonacci_braiding_circuit,
    run_quantum_gate_circuit
)
from app.analyze_results import analyze_simulation_results, display_experiment_summary
from app.scaling_analysis import display_scaling_analysis
from app.reference_tables import display_reference_tables

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
            params['scaling_factor'] = st.slider("Scaling Factor", 0.01, 2.00, PHI)
            params['n_steps'] = st.slider("Steps", 1, 100, 50)
            params['pulse_type'] = st.selectbox(
                "Pulse Type",
                ["Square", "Gaussian", "DRAG"]
            )
            
        elif mode == "Amplitude-Scaled Evolution":
            params['num_qubits'] = st.slider("Number of Qubits", 1, 4, 2)
            params['scaling_factor'] = st.slider("Amplitude Scale", 0.01, 2.00, 1.00)
            params['n_steps'] = st.slider("Steps", 1, 100, 50)
            params['hamiltonian_type'] = st.selectbox(
                "Hamiltonian",
                ["Ising", "Heisenberg", "Custom"]
            )
            params['use_phi_recursive'] = st.checkbox("Use Phi-Recursive Evolution", value=False)
            
            if params['use_phi_recursive']:
                params['recursion_depth'] = st.slider("Recursion Depth", 1, 5, 3)
                params['state_label'] = st.selectbox(
                    "Initial State",
                    ["plus", "phi_sensitive", "fractal", "fibonacci", "recursive"],
                    index=1  # "phi_sensitive" as default
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
                        scaling_factor=params['scaling_factor'],
                        noise_config=params.get('noise_config')
                    )
                elif mode == "Amplitude-Scaled Evolution":
                    if params.get('use_phi_recursive', False):
                        # Use phi-recursive evolution
                        result = run_phi_recursive_evolution(
                            num_qubits=params['num_qubits'],
                            state_label=params['state_label'],
                            n_steps=params['n_steps'],
                            scaling_factor=params['scaling_factor'],
                            recursion_depth=params['recursion_depth'],
                            analyze_phi=True
                        )
                    else:
                        # Use standard amplitude-scaled evolution
                        if params['num_qubits'] == 2:
                            # Use circuit-based implementation for 2 qubits
                            result = run_phi_scaled_twoqubit_circuit(
                                scaling_factor=params['scaling_factor'],
                                noise_config=params.get('noise_config')
                            )
                        else:
                            # Use state-based implementation for other qubit counts
                            result = run_state_evolution(
                                num_qubits=params['num_qubits'],
                                state_label="plus",
                                n_steps=params['n_steps'],
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
            
            except Exception:
                # Get the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                trace_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
                
                # Display error in Streamlit
                st.error("Simulation failed with the following error:")
                st.code(''.join(trace_details), language='python')
                
                # Also print to terminal for debugging
                print(''.join(trace_details), file=sys.stderr)
                return
    
    # Display results if available
    if st.session_state['simulation_results'] is not None:
        result = st.session_state['simulation_results']
        
    # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "State Evolution",
            "Noise Analysis",
            "Quantum Metrics",
            "Fractal Analysis",
            "Topological Analysis",
            "Scaling Analysis",
            "Dynamical Evolution",
            "Entanglement Dynamics",
            "Reference Tables",
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
                    title=f"{mode}"
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
            #TODO: add spinner or progress bar
            control_range = st.slider("Topological Control Parameter Range", 0.0, 10.0, (0.0, 5.0))
            from analyses.topology_plots import plot_invariants, plot_protection_metrics
            # Generate invariant plot using placeholder functions
            #TODO: replace with actual invariant computation
            fig_invariants = plot_invariants(control_range)
            st.pyplot(fig_invariants)
            # For demonstration, generate dummy protection metrics data
            #TODO: replace with actual protection metrics computation
            x_demo = np.linspace(control_range[0], control_range[1], 100)
            energy_gaps = np.abs(np.sin(x_demo))
            localization_measures = np.abs(np.cos(x_demo))
            fig_protection = plot_protection_metrics(control_range, energy_gaps, localization_measures)
            st.pyplot(fig_protection)
       
        # New Scaling Analysis tab
        with tab6:
            display_scaling_analysis(result, mode)
            
        # Dynamical Evolution tab
        with tab7:
            st.subheader("Wavepacket Evolution")
            if (hasattr(result, 'states') and hasattr(result, 'times') and 
                len(result.states) > 0 and len(result.times) > 0):
                
                # Create coordinates for wavepacket visualization
                if result.states[0].isket:
                    dim = len(result.states[0].full().flatten())
                else:
                    dim = result.states[0].shape[0]
                coordinates = np.linspace(0, 1, dim)
                
                # Plot wavepacket evolution
                from analyses.visualization.wavepacket_plots import plot_wavepacket_evolution, plot_wavepacket_spacetime
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Wavepacket Snapshots")
                    # Select time indices for snapshots
                    n_snapshots = min(6, len(result.states))
                    time_indices = np.linspace(0, len(result.states)-1, n_snapshots, dtype=int)
                    
                    # Plot wavepacket snapshots
                    fig_wavepacket = plot_wavepacket_evolution(
                        result.states,
                        result.times,
                        coordinates=coordinates,
                        time_indices=time_indices,
                        title=f"{mode} Wavepacket Evolution"
                    )
                    st.pyplot(fig_wavepacket)
                
                with col2:
                    st.write("Wavepacket Spacetime Diagram")
                    # Plot wavepacket spacetime diagram
                    fig_spacetime = plot_wavepacket_spacetime(
                        result.states,
                        result.times,
                        coordinates=coordinates,
                        title=f"{mode} Wavepacket Spacetime"
                    )
                    st.pyplot(fig_spacetime)
                
                # Add animation option
                if st.checkbox("Show Wavepacket Animation"):
                    from analyses.visualization.wavepacket_plots import animate_wavepacket_evolution
                    st.write("Wavepacket Evolution Animation")
                    
                    # Create animation
                    anim = animate_wavepacket_evolution(
                        result.states,
                        result.times,
                        coordinates=coordinates,
                        title=f"{mode} Wavepacket Animation"
                    )
                    
                    # Save animation to file
                    anim_path = "wavepacket_animation.gif"
                    anim.save(anim_path, writer='pillow', fps=10)
                    
                    # Display animation
                    st.image(anim_path)
            else:
                st.info("No evolution data available for wavepacket visualization.")
        
        # Entanglement Dynamics tab
        with tab8:
            st.subheader("Entanglement Dynamics")
            if (hasattr(result, 'states') and hasattr(result, 'times') and 
                len(result.states) > 1 and len(result.times) > 1):
                
                # Check if we have a multi-qubit system
                is_multipartite = False
                if result.states[0].isket:
                    num_qubits = len(result.states[0].dims[0])
                    is_multipartite = num_qubits > 1
                elif len(result.states[0].dims[0]) > 1:
                    num_qubits = len(result.states[0].dims[0])
                    is_multipartite = True
                
                if is_multipartite:
                    from analyses.entanglement_dynamics import (
                        plot_entanglement_entropy_vs_time,
                        plot_entanglement_spectrum,
                        plot_entanglement_growth_rate
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Entanglement Entropy Evolution")
                        # Plot entanglement entropy
                        fig_entropy = plot_entanglement_entropy_vs_time(
                            result.states,
                            result.times,
                            title=f"{mode} Entanglement Entropy"
                        )
                        st.pyplot(fig_entropy)
                    
                    with col2:
                        st.write("Entanglement Growth Rate")
                        # Plot entanglement growth rate
                        fig_growth = plot_entanglement_growth_rate(
                            result.states,
                            result.times,
                            title=f"{mode} Entanglement Growth"
                        )
                        st.pyplot(fig_growth)
                    
                    # Add entanglement spectrum
                    st.write("Entanglement Spectrum")
                    fig_spectrum = plot_entanglement_spectrum(
                        result.states,
                        result.times,
                        title=f"{mode} Entanglement Spectrum"
                    )
                    st.pyplot(fig_spectrum)
                else:
                    st.info("Entanglement analysis requires a multi-qubit system.")
            else:
                st.info("Entanglement analysis requires time evolution data.")
        
        # Reference Tables tab
        with tab9:
            display_reference_tables(result)
            
        # Export tab for simulation results
        with tab10:
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

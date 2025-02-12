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
import streamlit as st

# Local imports
from simulations.scripts.evolve_state import run_state_evolution
from simulations.scripts.evolve_circuit import (
    run_circuit_evolution,
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
                        n_steps=params['n_steps'],
                        scaling_factor=1.0
                    )
                elif mode == "State -> Phi-Scaled":
                    result = run_state_evolution(
                        num_qubits=params['num_qubits'],
                        state_label=params['state_label'],
                        n_steps=params['phi_steps'],
                        scaling_factor=params['scaling_factor']
                    )
                elif mode == "Circuit -> Standard 2Q":
                    result = run_circuit_evolution(
                        num_qubits=2,
                        scale_factor=1.0,
                        n_steps=50,
                        total_time=5.0
                    )
                elif mode == "Circuit -> Phi-Scaled 2Q":
                    result = run_circuit_evolution(
                        num_qubits=2,
                        scale_factor=1/PHI,
                        n_steps=50,
                        total_time=5.0
                    )
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
        tab1, tab2 = st.tabs(["Analysis", "Raw Data"])
        
        with tab1:
            analyze_simulation_results(result, mode)
        
        with tab2:
            display_experiment_summary(result, mode)
            
            # Option to download raw data
            if st.button("Download Raw Data"):
                # TODO: Implement data export functionality
                st.info("Data export functionality coming soon!")

if __name__ == "__main__":
    main()

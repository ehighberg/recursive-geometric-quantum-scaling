"""
Display reference tables for quantum simulations.
"""

import streamlit as st
import pandas as pd
from constants import PHI

from analyses.tables.parameter_tables import (
    generate_parameter_overview_table,
    generate_simulation_parameters_table,
    export_table_to_latex
)
from analyses.tables.phase_tables import (
    generate_phase_diagram_table,
    generate_phase_transition_table
)
from analyses.tables.performance_tables import (
    generate_performance_table,
    generate_convergence_table
)

def display_reference_tables(result=None):
    """
    Display reference tables for the quantum system.
    
    Parameters:
        result: Optional simulation result object
    """
    st.header("Reference Tables")
    
    # Create tabs for different table types
    tab1, tab2, tab3 = st.tabs(["Parameter Overview", "Phase Diagram", "Computational Performance"])
    
    with tab1:
        st.subheader("Parameter Overview")
        
        # Generate parameter overview table
        parameter_table = generate_parameter_overview_table()
        
        # Display table
        st.dataframe(parameter_table)
        
        # Add export buttons
        csv = parameter_table.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="parameter_overview.csv",
            mime="text/csv"
        )
        
        # Add LaTeX export
        latex_table = export_table_to_latex(
            parameter_table,
            "Comprehensive overview of model parameters.",
            "tab:parameters"
        )
        st.download_button(
            label="Download as LaTeX",
            data=latex_table,
            file_name="parameter_overview.tex",
            mime="text/plain"
        )
        
        # If result is available, show simulation parameters
        if result is not None:
            st.subheader("Simulation Parameters")
            sim_params_table = generate_simulation_parameters_table(result)
            st.dataframe(sim_params_table)
            
            # Add export button
            csv = sim_params_table.to_csv(index=False)
            st.download_button(
                label="Download Simulation Parameters as CSV",
                data=csv,
                file_name="simulation_parameters.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.subheader("Phase Diagram")
        
        # Options for phase diagram
        st.markdown("#### Phase Diagram Options")
        
        # Allow user to customize f_s ranges
        use_custom_ranges = st.checkbox("Use custom f_s ranges", value=False)
        
        if use_custom_ranges:
            # Let user define ranges
            num_ranges = st.number_input("Number of ranges", min_value=1, max_value=10, value=5)
            fs_ranges = []
            
            for i in range(num_ranges):
                col1, col2 = st.columns(2)
                with col1:
                    min_fs = st.number_input(f"Min f_s {i+1}", min_value=0.1, max_value=10.0, value=0.5 + i*0.5, step=0.1)
                with col2:
                    max_fs = st.number_input(f"Max f_s {i+1}", min_value=0.1, max_value=10.0, value=1.0 + i*0.5, step=0.1)
                fs_ranges.append((min_fs, max_fs))
        else:
            # Use default ranges
            fs_ranges = [
                (0.5, 1.0),
                (1.0, PHI-0.1),
                (PHI-0.1, PHI+0.1),
                (PHI+0.1, 2.0),
                (2.0, 3.0)
            ]
        
        # Generate phase diagram table
        # If we have results from scaling analysis, use them
        if 'scaling_results' in st.session_state:
            phase_table = generate_phase_diagram_table(
                fs_ranges=fs_ranges,
                results=st.session_state['scaling_results']
            )
        else:
            # Otherwise, generate template table
            phase_table = generate_phase_diagram_table(fs_ranges=fs_ranges)
        
        # Display table
        st.dataframe(phase_table)
        
        # Add export buttons
        csv = phase_table.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="phase_diagram.csv",
            mime="text/csv"
        )
        
        # Add LaTeX export
        latex_table = export_table_to_latex(
            phase_table,
            "Phase diagram summary showing phase types and properties across different f_s ranges.",
            "tab:phase_diagram"
        )
        st.download_button(
            label="Download as LaTeX",
            data=latex_table,
            file_name="phase_diagram.tex",
            mime="text/plain"
        )
        
        # Phase transitions table
        st.subheader("Phase Transitions")
        st.markdown("This table identifies potential phase transitions based on derivatives of quantum properties.")
        
        # If we have results from scaling analysis, generate phase transition table
        if 'scaling_results' in st.session_state:
            transition_table = generate_phase_transition_table(st.session_state['scaling_results'])
            st.dataframe(transition_table)
            
            # Add export button
            csv = transition_table.to_csv(index=False)
            st.download_button(
                label="Download Transitions as CSV",
                data=csv,
                file_name="phase_transitions.csv",
                mime="text/csv"
            )
        else:
            st.info("Run a scaling analysis to generate phase transition data.")
    
    with tab3:
        st.subheader("Computational Performance")
        
        # Generate performance table
        # If we have results with computation times, use them
        if result is not None and hasattr(result, 'computation_times'):
            performance_table = generate_performance_table(results=result)
            
            # Display table
            st.dataframe(performance_table)
            
            # Add export buttons
            csv = performance_table.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="computational_performance.csv",
                mime="text/csv"
            )
            
            # Add LaTeX export
            latex_table = export_table_to_latex(
                performance_table,
                "Computational performance metrics for different components of the simulation.",
                "tab:performance"
            )
            st.download_button(
                label="Download as LaTeX",
                data=latex_table,
                file_name="computational_performance.tex",
                mime="text/plain"
            )
            
            # Convergence table
            st.subheader("Convergence Metrics")
            convergence_table = generate_convergence_table(result)
            st.dataframe(convergence_table)
        else:
            # Otherwise, generate template table
            st.info("Run a simulation to generate performance data.")
            performance_table = generate_performance_table()
            st.dataframe(performance_table)
        
        # System size scaling section
        st.subheader("System Size Scaling")
        st.markdown("This table shows how computational resources scale with system size.")
        
        # Create template scaling table
        scaling_data = []
        for size in range(1, 5):
            scaling_data.append({
                "System Size (qubits)": size,
                "Hilbert Space Dimension": 2**size,
                "Theoretical Time Complexity": f"O(2^{2*size})",
                "Theoretical Memory": f"O(2^{size})"
            })
        
        scaling_table = pd.DataFrame(scaling_data)
        st.dataframe(scaling_table)
        
        # Add export button
        csv = scaling_table.to_csv(index=False)
        st.download_button(
            label="Download Scaling Table as CSV",
            data=csv,
            file_name="system_size_scaling.csv",
            mime="text/csv"
        )

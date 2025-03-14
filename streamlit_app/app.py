import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from constants import PHI
from simulations.scripts.evolve_state_fixed import (
    run_state_evolution_fixed,
    run_phi_recursive_evolution_fixed,
    run_comparative_analysis_fixed
)
from analyses.fractal_analysis_fixed import (
    calculate_fractal_dimension,
    compute_energy_spectrum
)
from analyses.visualization.fractal_plots import (
    plot_fractal_dimension_vs_recursion,
    plot_wavefunction_profile,
    plot_energy_spectrum
)
from analyses.visualization.wavepacket_plots import (
    plot_wavepacket_evolution,
    plot_wavepacket_spacetime
)
from analyses.entanglement_dynamics import (
    plot_entanglement_entropy,
    plot_entanglement_spectrum,
    plot_entanglement_growth_rate
)
from analyses.topological_invariants import (
    compute_topological_invariants,
    plot_topological_protection,
    plot_protection_ratio
)

# Set page configuration
st.set_page_config(
    page_title="Recursive Geometric Quantum Scaling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="main-header">Recursive Geometric Quantum Scaling</div>', unsafe_allow_html=True)
st.markdown("""
This interactive application allows you to explore the fascinating world of recursive geometric quantum scaling. 
You can simulate quantum systems with different scaling factors, visualize their behavior, and analyze the special 
properties that emerge at the golden ratio (φ ≈ 1.618034).
""")

# Sidebar for parameters
st.sidebar.markdown('<div class="sub-header">Simulation Parameters</div>', unsafe_allow_html=True)

# Scaling factor selection
scaling_factor_options = {
    "Golden Ratio (φ)": PHI,
    "Unit Scaling (1.0)": 1.0,
    "Custom Value": "custom"
}
scaling_factor_choice = st.sidebar.selectbox(
    "Select Scaling Factor",
    list(scaling_factor_options.keys())
)

if scaling_factor_choice == "Custom Value":
    scaling_factor = st.sidebar.number_input(
        "Enter Custom Scaling Factor",
        min_value=0.1,
        max_value=5.0,
        value=1.5,
        step=0.1
    )
else:
    scaling_factor = scaling_factor_options[scaling_factor_choice]

# System size
system_size = st.sidebar.slider(
    "System Size",
    min_value=4,
    max_value=16,
    value=8,
    step=2
)

# Recursion depth
recursion_depth = st.sidebar.slider(
    "Recursion Depth",
    min_value=1,
    max_value=5,
    value=3
)

# Time steps
time_steps = st.sidebar.slider(
    "Time Steps",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Fractal Structure", "Wavepacket Evolution", "Entanglement Dynamics", "Topological Protection", "Comparative Analysis"]
)

# Run simulation button
run_simulation = st.sidebar.button("Run Simulation")

# Main content area
if run_simulation:
    st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    if analysis_type == "Fractal Structure":
        st.markdown("### Fractal Structure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Energy Spectrum")
            progress_bar.progress(25)
            
            # Compute and plot energy spectrum
            spectrum_result = compute_energy_spectrum(
                system_size=system_size,
                scaling_factor=scaling_factor,
                recursion_depth=recursion_depth
            )
            
            fig_spectrum = plot_energy_spectrum(spectrum_result)
            st.pyplot(fig_spectrum)
            
        with col2:
            st.markdown("#### Wavefunction Profile")
            progress_bar.progress(50)
            
            # Plot wavefunction profile
            fig_wavefunction = plot_wavefunction_profile(
                system_size=system_size,
                scaling_factor=scaling_factor,
                recursion_depth=recursion_depth
            )
            st.pyplot(fig_wavefunction)
        
        st.markdown("#### Fractal Dimension vs. Recursion Depth")
        progress_bar.progress(75)
        
        # Plot fractal dimension vs recursion
        fig_fractal = plot_fractal_dimension_vs_recursion(
            max_recursion=5,
            system_size=system_size,
            scaling_factors=[1.0, scaling_factor, PHI]
        )
        st.pyplot(fig_fractal)
        
        # Display fractal dimension
        fractal_dim = calculate_fractal_dimension(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        
        st.markdown(f"""
        <div class="highlight info-text">
        <b>Fractal Dimension:</b> {fractal_dim:.4f}
        </div>
        """, unsafe_allow_html=True)
        
    elif analysis_type == "Wavepacket Evolution":
        st.markdown("### Wavepacket Evolution Analysis")
        
        # Run state evolution
        progress_bar.progress(25)
        evolution_result = run_state_evolution_fixed(
            num_qubits=system_size,
            scaling_factor=scaling_factor,
            n_steps=time_steps
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Wavepacket Evolution")
            progress_bar.progress(50)
            
            # Plot wavepacket evolution
            fig_evolution = plot_wavepacket_evolution(evolution_result)
            st.pyplot(fig_evolution)
            
        with col2:
            st.markdown("#### Spacetime Diagram")
            progress_bar.progress(75)
            
            # Plot spacetime diagram
            fig_spacetime = plot_wavepacket_spacetime(evolution_result)
            st.pyplot(fig_spacetime)
        
        # Display propagation velocity
        velocity = evolution_result.propagation_velocity
        
        st.markdown(f"""
        <div class="highlight info-text">
        <b>Propagation Velocity:</b> {velocity:.4f}
        </div>
        """, unsafe_allow_html=True)
        
    elif analysis_type == "Entanglement Dynamics":
        st.markdown("### Entanglement Dynamics Analysis")
        
        # Run state evolution
        progress_bar.progress(25)
        evolution_result = run_state_evolution_fixed(
            num_qubits=system_size,
            scaling_factor=scaling_factor,
            n_steps=time_steps
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Entanglement Entropy")
            progress_bar.progress(50)
            
            # Plot entanglement entropy
            fig_entropy = plot_entanglement_entropy(evolution_result)
            st.pyplot(fig_entropy)
            
        with col2:
            st.markdown("#### Entanglement Spectrum")
            progress_bar.progress(75)
            
            # Plot entanglement spectrum
            fig_spectrum = plot_entanglement_spectrum(evolution_result)
            st.pyplot(fig_spectrum)
        
        st.markdown("#### Entanglement Growth Rate")
        
        # Plot entanglement growth rate
        fig_growth = plot_entanglement_growth_rate(evolution_result)
        st.pyplot(fig_growth)
        
        # Display max entanglement
        max_entanglement = max(evolution_result.entanglement_entropy)
        
        st.markdown(f"""
        <div class="highlight info-text">
        <b>Maximum Entanglement Entropy:</b> {max_entanglement:.4f}
        </div>
        """, unsafe_allow_html=True)
        
    elif analysis_type == "Topological Protection":
        st.markdown("### Topological Protection Analysis")
        
        # Compute topological invariants
        progress_bar.progress(25)
        topo_result = compute_topological_invariants(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Robustness Under Perturbations")
            progress_bar.progress(50)
            
            # Plot topological protection
            fig_protection = plot_topological_protection(
                system_size=system_size,
                scaling_factors=[1.0, scaling_factor, PHI]
            )
            st.pyplot(fig_protection)
            
        with col2:
            st.markdown("#### Protection Ratio")
            progress_bar.progress(75)
            
            # Plot protection ratio
            fig_ratio = plot_protection_ratio(
                system_size=system_size,
                scaling_factors=np.linspace(0.5, 3.0, 26)
            )
            st.pyplot(fig_ratio)
        
        # Display topological invariant
        invariant = topo_result.topological_invariant
        
        st.markdown(f"""
        <div class="highlight info-text">
        <b>Topological Invariant:</b> {invariant:.4f}
        </div>
        """, unsafe_allow_html=True)
        
    elif analysis_type == "Comparative Analysis":
        st.markdown("### Comparative Analysis")
        
        # Run comparative analysis
        progress_bar.progress(25)
        scaling_factors = [1.0, scaling_factor, PHI]
        
        # Run the comparative analysis with all scaling factors
        comparative_results = run_comparative_analysis_fixed(
            scaling_factors=scaling_factors,
            num_qubits=system_size,
            n_steps=time_steps,
            recursion_depth=recursion_depth
        )
        
        # Extract results for each scaling factor
        results = []
        for sf in scaling_factors:
            # Get standard and phi-recursive results
            std_result = comparative_results['standard_results'][sf]
            phi_result = comparative_results['phi_recursive_results'][sf]
            metrics = comparative_results['comparative_metrics'][sf]
            
            # Combine into a single result object for display
            result = {
                'fractal_dimension': phi_result.phi_dimension if hasattr(phi_result, 'phi_dimension') and phi_result.phi_dimension is not None else 0.0,
                'propagation_velocity': getattr(std_result, 'propagation_velocity', 0.0),
                'entanglement_entropy': getattr(std_result, 'entanglement_entropy', [0.0]),
                'topological_protection': metrics.get('state_overlap', 0.0)
            }
            results.append(result)
        
        progress_bar.progress(50)
        
        # Create comparison table
        comparison_data = {
            "Scaling Factor": ["Unit (1.0)", f"Selected ({scaling_factor:.6f})", f"Golden Ratio ({PHI:.6f})"],
            "Fractal Dimension": [
                f"{results[0].fractal_dimension:.4f}",
                f"{results[1].fractal_dimension:.4f}",
                f"{results[2].fractal_dimension:.4f}"
            ],
            "Propagation Velocity": [
                f"{results[0].propagation_velocity:.4f}",
                f"{results[1].propagation_velocity:.4f}",
                f"{results[2].propagation_velocity:.4f}"
            ],
            "Max Entanglement": [
                f"{max(results[0].entanglement_entropy):.4f}",
                f"{max(results[1].entanglement_entropy):.4f}",
                f"{max(results[2].entanglement_entropy):.4f}"
            ],
            "Topological Protection": [
                f"{results[0].topological_protection:.4f}",
                f"{results[1].topological_protection:.4f}",
                f"{results[2].topological_protection:.4f}"
            ]
        }
        
        st.table(comparison_data)
        
        progress_bar.progress(75)
        
        # Plot comparison of wavepacket evolution
        st.markdown("#### Wavepacket Evolution Comparison")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (sf, result) in enumerate(zip(scaling_factors, results)):
            sf_label = "Unit" if sf == 1.0 else ("Phi" if sf == PHI else f"{sf:.2f}")
            axes[i].imshow(
                result.wavepacket_evolution.T,
                aspect='auto',
                cmap='viridis',
                origin='lower'
            )
            axes[i].set_title(f"Scaling Factor: {sf_label}")
            axes[i].set_xlabel("Position")
            axes[i].set_ylabel("Time")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Complete progress bar
    progress_bar.progress(100)

# Footer
st.markdown("""
<div class="footer">
Recursive Geometric Quantum Scaling (RGQS) - Interactive Simulation Tool
</div>
""", unsafe_allow_html=True)

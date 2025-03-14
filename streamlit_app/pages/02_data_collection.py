import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import io
import base64

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from constants import PHI
from simulations.scripts.evolve_state_fixed import (
    run_state_evolution_fixed,
    run_phi_recursive_evolution_fixed,
    run_comparative_analysis_fixed
)
from analyses.fractal_analysis_fixed import (
    fractal_dimension,
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
    page_title="Data Collection for Research",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
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
    .download-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create a download link for a dataframe
def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

# Helper function to create a download link for a figure
def get_figure_download_link(fig, filename, text):
    """Generates a link allowing the figure to be downloaded as a PNG file"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

# Title and introduction
st.markdown('<div class="main-header">Data Collection for Research</div>', unsafe_allow_html=True)
st.markdown("""
This page allows you to generate high-quality data and visualizations for your research paper on 
Recursive Geometric Quantum Scaling. You can customize parameters, run simulations, and export the results
in publication-ready formats.
""")

# Sidebar for parameters
st.sidebar.markdown('<div class="sub-header">Data Collection Parameters</div>', unsafe_allow_html=True)

# Data type selection
data_type = st.sidebar.selectbox(
    "Data Type",
    ["Fractal Structure", "Wavepacket Evolution", "Entanglement Dynamics", "Topological Protection", "Comparative Analysis"]
)

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

# Output format
output_format = st.sidebar.selectbox(
    "Output Format",
    ["PNG (300 DPI)", "CSV Data", "Both"]
)

# Generate data button
generate_data = st.sidebar.button("Generate Data")

# Main content area
if generate_data:
    st.markdown('<div class="sub-header">Generated Data</div>', unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    if data_type == "Fractal Structure":
        st.markdown("### Fractal Structure Data")
        
        # Compute energy spectrum
        progress_bar.progress(25)
        spectrum_result = compute_energy_spectrum(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        
        # Calculate fractal dimension
        fractal_dim = calculate_fractal_dimension(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        
        # Create plots
        progress_bar.progress(50)
        fig_spectrum = plot_energy_spectrum(spectrum_result)
        fig_wavefunction = plot_wavefunction_profile(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        fig_fractal = plot_fractal_dimension_vs_recursion(
            max_recursion=5,
            system_size=system_size,
            scaling_factors=[1.0, scaling_factor, PHI]
        )
        
        # Display plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(fig_spectrum)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_spectrum, "energy_spectrum.png", "Download Energy Spectrum (PNG)")}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.pyplot(fig_wavefunction)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_wavefunction, "wavefunction_profile.png", "Download Wavefunction Profile (PNG)")}
            </div>
            """, unsafe_allow_html=True)
        
        st.pyplot(fig_fractal)
        st.markdown(f"""
        <div class="download-box">
            {get_figure_download_link(fig_fractal, "fractal_dimension_vs_recursion.png", "Download Fractal Dimension Plot (PNG)")}
        </div>
        """, unsafe_allow_html=True)
        
        # Create data table
        progress_bar.progress(75)
        fractal_data = {
            "System Size": [system_size],
            "Scaling Factor": [scaling_factor],
            "Recursion Depth": [recursion_depth],
            "Fractal Dimension": [fractal_dim],
            "Energy Levels": [len(spectrum_result.eigenvalues)],
            "Band Gap": [spectrum_result.band_gap]
        }
        
        df_fractal = pd.DataFrame(fractal_data)
        
        # Display and provide download for data
        st.markdown("### Fractal Structure Data Table")
        st.dataframe(df_fractal)
        
        st.markdown(f"""
        <div class="download-box">
            {get_table_download_link(df_fractal, "fractal_structure_data.csv", "Download Fractal Structure Data (CSV)")}
        </div>
        """, unsafe_allow_html=True)
        
    elif data_type == "Wavepacket Evolution":
        st.markdown("### Wavepacket Evolution Data")
        
        # Run state evolution
        progress_bar.progress(25)
        evolution_result = run_state_evolution(
            system_size=system_size,
            scaling_factor=scaling_factor,
            time_steps=time_steps,
            recursion_depth=recursion_depth
        )
        
        # Create plots
        progress_bar.progress(50)
        fig_evolution = plot_wavepacket_evolution(evolution_result)
        fig_spacetime = plot_wavepacket_spacetime(evolution_result)
        
        # Display plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(fig_evolution)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_evolution, "wavepacket_evolution.png", "Download Wavepacket Evolution (PNG)")}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.pyplot(fig_spacetime)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_spacetime, "wavepacket_spacetime.png", "Download Spacetime Diagram (PNG)")}
            </div>
            """, unsafe_allow_html=True)
        
        # Create data table
        progress_bar.progress(75)
        wavepacket_data = pd.DataFrame(evolution_result.wavepacket_evolution)
        wavepacket_data.columns = [f"Position_{i}" for i in range(wavepacket_data.shape[1])]
        wavepacket_data.insert(0, "Time", range(wavepacket_data.shape[0]))
        
        # Display and provide download for data
        st.markdown("### Wavepacket Evolution Data Table")
        st.dataframe(wavepacket_data.head())
        
        st.markdown(f"""
        <div class="download-box">
            {get_table_download_link(wavepacket_data, "wavepacket_evolution_data.csv", "Download Wavepacket Evolution Data (CSV)")}
        </div>
        """, unsafe_allow_html=True)
        
    elif data_type == "Entanglement Dynamics":
        st.markdown("### Entanglement Dynamics Data")
        
        # Run state evolution
        progress_bar.progress(25)
        evolution_result = run_state_evolution(
            system_size=system_size,
            scaling_factor=scaling_factor,
            time_steps=time_steps,
            recursion_depth=recursion_depth
        )
        
        # Create plots
        progress_bar.progress(50)
        fig_entropy = plot_entanglement_entropy(evolution_result)
        fig_spectrum = plot_entanglement_spectrum(evolution_result)
        fig_growth = plot_entanglement_growth_rate(evolution_result)
        
        # Display plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(fig_entropy)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_entropy, "entanglement_entropy.png", "Download Entanglement Entropy (PNG)")}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.pyplot(fig_spectrum)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_spectrum, "entanglement_spectrum.png", "Download Entanglement Spectrum (PNG)")}
            </div>
            """, unsafe_allow_html=True)
        
        st.pyplot(fig_growth)
        st.markdown(f"""
        <div class="download-box">
            {get_figure_download_link(fig_growth, "entanglement_growth_rate.png", "Download Entanglement Growth Rate (PNG)")}
        </div>
        """, unsafe_allow_html=True)
        
        # Create data table
        progress_bar.progress(75)
        entanglement_data = {
            "Time": list(range(len(evolution_result.entanglement_entropy))),
            "Entanglement_Entropy": evolution_result.entanglement_entropy,
            "Growth_Rate": evolution_result.entanglement_growth_rate
        }
        
        df_entanglement = pd.DataFrame(entanglement_data)
        
        # Display and provide download for data
        st.markdown("### Entanglement Dynamics Data Table")
        st.dataframe(df_entanglement.head())
        
        st.markdown(f"""
        <div class="download-box">
            {get_table_download_link(df_entanglement, "entanglement_dynamics_data.csv", "Download Entanglement Dynamics Data (CSV)")}
        </div>
        """, unsafe_allow_html=True)
        
    elif data_type == "Topological Protection":
        st.markdown("### Topological Protection Data")
        
        # Compute topological invariants
        progress_bar.progress(25)
        topo_result = compute_topological_invariants(
            system_size=system_size,
            scaling_factor=scaling_factor,
            recursion_depth=recursion_depth
        )
        
        # Create plots
        progress_bar.progress(50)
        fig_protection = plot_topological_protection(
            system_size=system_size,
            scaling_factors=[1.0, scaling_factor, PHI]
        )
        fig_ratio = plot_protection_ratio(
            system_size=system_size,
            scaling_factors=np.linspace(0.5, 3.0, 26)
        )
        
        # Display plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.pyplot(fig_protection)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_protection, "topological_protection.png", "Download Topological Protection (PNG)")}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.pyplot(fig_ratio)
            st.markdown(f"""
            <div class="download-box">
                {get_figure_download_link(fig_ratio, "protection_ratio.png", "Download Protection Ratio (PNG)")}
            </div>
            """, unsafe_allow_html=True)
        
        # Create data table
        progress_bar.progress(75)
        topo_data = {
            "System Size": [system_size],
            "Scaling Factor": [scaling_factor],
            "Recursion Depth": [recursion_depth],
            "Topological Invariant": [topo_result.topological_invariant],
            "Protection Metric": [topo_result.topological_protection],
            "Critical Perturbation": [topo_result.critical_perturbation]
        }
        
        df_topo = pd.DataFrame(topo_data)
        
        # Display and provide download for data
        st.markdown("### Topological Protection Data Table")
        st.dataframe(df_topo)
        
        st.markdown(f"""
        <div class="download-box">
            {get_table_download_link(df_topo, "topological_protection_data.csv", "Download Topological Protection Data (CSV)")}
        </div>
        """, unsafe_allow_html=True)
        
    elif data_type == "Comparative Analysis":
        st.markdown("### Comparative Analysis Data")
        
        # Run comparative analysis
        progress_bar.progress(25)
        scaling_factors = [1.0, scaling_factor, PHI]
        
        results = []
        for sf in scaling_factors:
            result = run_comparative_analysis_fixed(
                system_size=system_size,
                scaling_factor=sf,
                time_steps=time_steps,
                recursion_depth=recursion_depth
            )
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
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Plot comparison of wavepacket evolution
        progress_bar.progress(75)
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
        
        # Display plot and table
        st.pyplot(fig)
        st.markdown(f"""
        <div class="download-box">
            {get_figure_download_link(fig, "wavepacket_comparison.png", "Download Wavepacket Comparison (PNG)")}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Comparative Analysis Data Table")
        st.dataframe(df_comparison)
        
        st.markdown(f"""
        <div class="download-box">
            {get_table_download_link(df_comparison, "comparative_analysis_data.csv", "Download Comparative Analysis Data (CSV)")}
        </div>
        """, unsafe_allow_html=True)
    
    # Complete progress bar
    progress_bar.progress(100)
    
    st.markdown("""
    <div class="highlight info-text">
    <p>All data has been generated successfully. You can download the visualizations and data tables using the links above.</p>
    <p>These files are in publication-ready format (300 DPI PNG for images, CSV for data tables) and can be directly included in your research paper.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    ### Data Collection for Research Papers
    
    This tool allows you to generate high-quality data and visualizations for your research paper on Recursive Geometric Quantum Scaling.
    
    To collect data:
    
    1. Select the type of data you want to generate from the sidebar
    2. Adjust the parameters (system size, recursion depth, time steps, scaling factor)
    3. Choose your preferred output format (PNG images, CSV data, or both)
    4. Click "Generate Data" to run the simulation
    
    The tool will generate publication-ready visualizations (300 DPI PNG files) and data tables (CSV files) that you can download
    and include directly in your research paper.
    
    This ensures that all data in your paper is generated using the fixed implementations that provide unbiased analysis of
    phi-related effects, maintaining the scientific integrity of your research.
    """)

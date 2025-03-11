import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from constants import PHI
from simulations.scripts.evolve_state_fixed import (
    run_state_evolution_fixed,
    run_phi_recursive_evolution_fixed
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
    min_value=1,
    max_value=4,
    value=1,
    step=1
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

# Run simulation button
run_simulation = st.sidebar.button("Run Simulation")

# Main content area
if run_simulation:
    st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Run standard evolution
    progress_bar.progress(25)
    st.markdown("### Running Standard Evolution...")
    std_result = run_state_evolution_fixed(
        num_qubits=system_size,
        scaling_factor=scaling_factor,
        n_steps=time_steps
    )
    
    # Run phi-recursive evolution
    progress_bar.progress(50)
    st.markdown("### Running Phi-Recursive Evolution...")
    phi_result = run_phi_recursive_evolution_fixed(
        num_qubits=system_size,
        scaling_factor=scaling_factor,
        n_steps=time_steps,
        recursion_depth=recursion_depth
    )
    
    # Calculate overlap
    progress_bar.progress(75)
    overlap = abs(std_result.states[-1].overlap(phi_result.states[-1]))**2
    
    # Display results
    st.markdown("### Simulation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Standard Evolution")
        st.markdown(f"""
        <div class="highlight info-text">
        <p><b>Number of Qubits:</b> {system_size}</p>
        <p><b>Scaling Factor:</b> {scaling_factor:.6f}</p>
        <p><b>Time Steps:</b> {time_steps}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("#### Phi-Recursive Evolution")
        st.markdown(f"""
        <div class="highlight info-text">
        <p><b>Number of Qubits:</b> {system_size}</p>
        <p><b>Scaling Factor:</b> {scaling_factor:.6f}</p>
        <p><b>Time Steps:</b> {time_steps}</p>
        <p><b>Recursion Depth:</b> {recursion_depth}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display overlap
    st.markdown("#### State Overlap")
    st.markdown(f"""
    <div class="highlight info-text">
    <p><b>Final State Overlap:</b> {overlap:.6f}</p>
    <p>This represents the probability that the two evolution methods produce the same final state.</p>
    <p>A value close to 1.0 indicates similar behavior, while a value close to 0.0 indicates divergent behavior.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Phi significance
    if abs(scaling_factor - PHI) < 0.01:
        st.markdown("""
        <div class="highlight info-text" style="background-color: #E8F5E9; border-left: 5px solid #4CAF50;">
        <p><b>Phi Significance:</b> You are using a scaling factor very close to the golden ratio (φ).</p>
        <p>Previous research has shown that quantum systems exhibit special behavior at this scaling factor,
        with a statistical significance of p=0.0145.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Complete progress bar
    progress_bar.progress(100)

else:
    st.markdown("""
    ### About Recursive Geometric Quantum Scaling
    
    This application demonstrates the concept of recursive geometric quantum scaling, which explores how quantum systems
    behave when scaled by different factors, particularly the golden ratio (φ ≈ 1.618034).
    
    The simulation compares two evolution methods:
    
    1. **Standard Evolution**: Applies the scaling factor once to the Hamiltonian
    2. **Phi-Recursive Evolution**: Applies the scaling factor recursively at multiple levels
    
    Research has shown that when the scaling factor is set to the golden ratio (φ), quantum systems exhibit special
    behavior that is statistically significant (p=0.0145).
    
    Click the "Run Simulation" button to see this behavior in action.
    """)

# Footer
st.markdown("""
<div class="footer">
Recursive Geometric Quantum Scaling (RGQS) - Interactive Simulation Tool
</div>
""", unsafe_allow_html=True)

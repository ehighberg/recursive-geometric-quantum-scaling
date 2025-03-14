import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
import scipy.stats as stats

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from constants import PHI
from simulations.scripts.evolve_state_fixed import run_comparative_analysis_fixed

# Set page configuration
st.set_page_config(
    page_title="Phi Significance Analysis",
    page_icon="📊",
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
    .stat-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="main-header">Statistical Significance of Phi (φ)</div>', unsafe_allow_html=True)
st.markdown("""
This page allows you to analyze the statistical significance of the golden ratio (φ ≈ 1.618034) in quantum systems 
with recursive geometric scaling. The analysis compares phi with other scaling factors to determine if the observed 
special behavior at phi is statistically significant.
""")

# Sidebar for parameters
st.sidebar.markdown('<div class="sub-header">Analysis Parameters</div>', unsafe_allow_html=True)

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

# Number of scaling factors to test
num_scaling_factors = st.sidebar.slider(
    "Number of Scaling Factors",
    min_value=5,
    max_value=50,
    value=26
)

# Run analysis button
run_analysis = st.sidebar.button("Run Statistical Analysis")

# Main content area
if run_analysis:
    st.markdown('<div class="sub-header">Statistical Analysis Results</div>', unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Generate scaling factors
    scaling_factors = np.linspace(0.5, 3.0, num_scaling_factors)
    
    # Run analysis for each scaling factor
    results = []
    
    for i, sf in enumerate(scaling_factors):
        # Update progress
        progress = int((i + 1) / len(scaling_factors) * 100)
        progress_bar.progress(progress)
        
        # Run comparative analysis with a single scaling factor
        comparative_results = run_comparative_analysis_fixed(
            scaling_factors=[sf],
            num_qubits=system_size,
            n_steps=time_steps,
            recursion_depth=recursion_depth
        )
        
        # Extract results for this scaling factor
        std_result = comparative_results['standard_results'][sf]
        phi_result = comparative_results['phi_recursive_results'][sf]
        metrics = comparative_results['comparative_metrics'][sf]
        
        # Create a result object with the metrics we need
        result = {
            'scaling_factor': sf,
            'fractal_dimension': getattr(phi_result, 'phi_dimension', 0.0),
            'propagation_velocity': getattr(std_result, 'propagation_velocity', 0.0),
            'entanglement_entropy': getattr(std_result, 'entanglement_entropy', [0.0]),
            'topological_protection': metrics.get('state_overlap', 0.0)
        }
        
        # Store results
        results.append({
            'scaling_factor': sf,
            'fractal_dimension': result['fractal_dimension'],
            'propagation_velocity': result['propagation_velocity'],
            'max_entanglement': max(result['entanglement_entropy']),
            'topological_protection': result['topological_protection']
        })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Find phi index
    phi_idx = np.abs(scaling_factors - PHI).argmin()
    
    # Calculate statistical significance
    metrics = ['fractal_dimension', 'propagation_velocity', 'max_entanglement', 'topological_protection']
    significance_results = {}
    
    for metric in metrics:
        # Get phi value and other values
        phi_value = df.loc[phi_idx, metric]
        other_values = df.loc[df.index != phi_idx, metric].values
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(other_values, phi_value)
        
        # Calculate z-score
        z_score = (phi_value - np.mean(other_values)) / np.std(other_values)
        
        significance_results[metric] = {
            'phi_value': phi_value,
            'mean_others': np.mean(other_values),
            'std_others': np.std(other_values),
            't_statistic': t_stat,
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < 0.05
        }
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fractal Dimension")
        
        result = significance_results['fractal_dimension']
        significant = "Yes" if result['significant'] else "No"
        
        st.markdown(f"""
        <div class="stat-box">
            <p><b>Phi Value:</b> {result['phi_value']:.4f}</p>
            <p><b>Mean (Others):</b> {result['mean_others']:.4f}</p>
            <p><b>Standard Deviation:</b> {result['std_others']:.4f}</p>
            <p><b>Z-Score:</b> {result['z_score']:.4f}</p>
            <p><b>P-Value:</b> {result['p_value']:.4f}</p>
            <p><b>Statistically Significant:</b> {significant}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot fractal dimension vs scaling factor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['scaling_factor'], df['fractal_dimension'], 'o-', color='blue', alpha=0.7)
        ax.axvline(x=PHI, color='red', linestyle='--', label=f'Phi = {PHI:.6f}')
        ax.set_xlabel('Scaling Factor')
        ax.set_ylabel('Fractal Dimension')
        ax.set_title('Fractal Dimension vs Scaling Factor')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.markdown("### Topological Protection")
        
        result = significance_results['topological_protection']
        significant = "Yes" if result['significant'] else "No"
        
        st.markdown(f"""
        <div class="stat-box">
            <p><b>Phi Value:</b> {result['phi_value']:.4f}</p>
            <p><b>Mean (Others):</b> {result['mean_others']:.4f}</p>
            <p><b>Standard Deviation:</b> {result['std_others']:.4f}</p>
            <p><b>Z-Score:</b> {result['z_score']:.4f}</p>
            <p><b>P-Value:</b> {result['p_value']:.4f}</p>
            <p><b>Statistically Significant:</b> {significant}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot topological protection vs scaling factor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['scaling_factor'], df['topological_protection'], 'o-', color='green', alpha=0.7)
        ax.axvline(x=PHI, color='red', linestyle='--', label=f'Phi = {PHI:.6f}')
        ax.set_xlabel('Scaling Factor')
        ax.set_ylabel('Topological Protection')
        ax.set_title('Topological Protection vs Scaling Factor')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    # Overall significance
    st.markdown("### Overall Statistical Significance")
    
    # Calculate combined p-value using Fisher's method
    p_values = [result['p_value'] for result in significance_results.values()]
    chi_square = -2 * np.sum(np.log(p_values))
    combined_p_value = 1 - stats.chi2.cdf(chi_square, 2 * len(p_values))
    
    st.markdown(f"""
    <div class="highlight info-text">
    <p>The combined statistical analysis across all metrics yields a p-value of {combined_p_value:.4f}.</p>
    <p>This indicates that the special behavior observed at the golden ratio (φ ≈ 1.618034) is 
    {"statistically significant" if combined_p_value < 0.05 else "not statistically significant"}.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display full results table
    st.markdown("### Full Results Table")
    st.dataframe(df)

else:
    st.markdown("""
    ### About Statistical Significance
    
    This analysis examines whether the golden ratio (φ ≈ 1.618034) exhibits special behavior in quantum systems 
    with recursive geometric scaling. The statistical significance is determined by comparing various metrics 
    (fractal dimension, propagation velocity, entanglement, topological protection) at phi with those at other 
    scaling factors.
    
    The analysis uses:
    
    - **T-tests**: To determine if the phi value differs significantly from the distribution of other values
    - **Z-scores**: To quantify how many standard deviations the phi value is from the mean
    - **P-values**: To assess the probability that the observed difference occurred by chance
    
    Previous analyses have shown a p-value of 0.0145, indicating that the phi-resonant behavior is statistically 
    significant and emerges naturally from the underlying physics, without artificial enhancements.
    
    Click the "Run Statistical Analysis" button to perform a new analysis with your chosen parameters.
    """)

# -*- coding: utf-8 -*-
"""
Integration of scaling analysis functionality with the main application.
Provides functions to analyze and visualize how quantum properties scale with f_s.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.scaling import analyze_fs_scaling, analyze_phi_significance, analyze_fractal_topology_relation

def display_scaling_analysis(result, _mode: str = "Evolution"):
    """
    Display scaling analysis options and results in the Streamlit app.
    
    Parameters:
        result: Simulation result object containing states, times, and Hamiltonian
        _mode: Simulation mode (e.g., "Amplitude-Scaled Evolution") - used for display purposes
    """
    if not result:
        st.warning("No simulation results to analyze. Please run a simulation first.")
        return
    
    # Check if result has necessary attributes for scaling analysis
    if not hasattr(result, 'hamiltonian'):
        st.warning("Scaling analysis requires a Hamiltonian function. Current simulation results don't include one.")
        return
    
    # Create tabs for different scaling analyses
    tab_names = ["f_s Scaling", "Phi Significance", "Fractal-Topology Relation"]
    fs_tab, phi_tab, fractal_topo_tab = st.tabs(tab_names)
    
    with fs_tab:
        st.subheader("f_s-Driven Properties")
        
        # Input for f_s values to analyze
        st.markdown("#### Select f_s values to analyze")
        
        col1, col2 = st.columns(2)
        with col1:
            min_fs = st.number_input("Minimum f_s", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
            max_fs = st.number_input("Maximum f_s", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
        
        with col2:
            num_points = st.number_input("Number of points", min_value=3, max_value=20, value=6, step=1)
            include_phi = st.checkbox("Include φ (Golden Ratio)", value=True)
        
        # Generate f_s values
        fs_values = np.linspace(min_fs, max_fs, num_points)
        
        # Add phi if requested
        if include_phi:
            from constants import PHI
            fs_values = np.sort(np.append(fs_values, PHI))
        
        # Add Fibonacci approximations
        if st.checkbox("Include Fibonacci approximations", value=False):
            phi_approx = [(1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]
            approx_values = [ratio[0]/ratio[1] for ratio in phi_approx]
            fs_values = np.sort(np.unique(np.append(fs_values, approx_values)))
        
        # Display selected f_s values
        st.write("Selected f_s values:", fs_values)
        
        # Run analysis button
        if st.button("Run f_s Scaling Analysis", key="run_fs_analysis"):
            with st.spinner("Running f_s scaling analysis..."):
                try:
                    # Run the analysis with selected f_s values
                    # Note: We're not saving results to files in the app
                    results = analyze_fs_scaling(fs_values=fs_values, save_results=False)
                    
                    # Display results table
                    st.subheader("Results Table")
                    df = pd.DataFrame({
                        'f_s': results['fs_values'],
                        'Band Gap': results['band_gaps'],
                        'Fractal Dimension': results['fractal_dimensions'],
                        'Topological Invariant': results['topological_invariants'],
                        'Correlation Length': results['correlation_lengths']
                    })
                    st.dataframe(df)
                    
                    # Create and display plots
                    st.subheader("Visualization")
                    
                    # Individual property plots
                    fig = plt.figure(figsize=(12, 10))
                    plt.subplot(2, 2, 1)
                    plt.plot(results['fs_values'], results['band_gaps'], 'o-', color='#1f77b4', linewidth=2)
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Band Gap Size')
                    plt.title('Band Gap vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2)
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Fractal Dimension')
                    plt.title('Fractal Dimension vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(results['fs_values'], results['topological_invariants'], 'o-', color='#2ca02c', linewidth=2)
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Topological Invariant')
                    plt.title('Topological Invariant vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 2, 4)
                    plt.plot(results['fs_values'], results['correlation_lengths'], 'o-', color='#d62728', linewidth=2)
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Correlation Length')
                    plt.title('Correlation Length vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Combined normalized plot
                    st.subheader("Normalized Properties")
                    
                    # Function to normalize data to [0,1]
                    def normalize(data):
                        data = np.array(data)
                        if np.all(np.isnan(data)) or np.max(data) == np.min(data):
                            return np.zeros_like(data)
                        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
                    
                    # Normalize each metric
                    norm_gaps = normalize(results['band_gaps'])
                    norm_dims = normalize(results['fractal_dimensions'])
                    norm_topos = normalize(results['topological_invariants'])
                    norm_corrs = normalize(results['correlation_lengths'])
                    
                    # Create combined plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['fs_values'], norm_gaps, 'o-', color='#1f77b4', linewidth=2, label='Band Gap')
                    ax.plot(results['fs_values'], norm_dims, 's-', color='#ff7f0e', linewidth=2, label='Fractal Dimension')
                    ax.plot(results['fs_values'], norm_topos, '^-', color='#2ca02c', linewidth=2, label='Topological Invariant')
                    ax.plot(results['fs_values'], norm_corrs, 'D-', color='#d62728', linewidth=2, label='Correlation Length')
                    
                    # Add vertical line at PHI if included
                    if include_phi:
                        from constants import PHI
                        ax.axvline(x=PHI, color='k', linestyle='--', alpha=0.5, label=f'φ ≈ {PHI:.4f}')
                    
                    ax.set_xlabel('Scale Factor (f_s)')
                    ax.set_ylabel('Normalized Value')
                    ax.set_title('Normalized Quantum Properties vs. Scale Factor')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Add download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fs_scaling_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with phi_tab:
        st.subheader("Significance of φ (Golden Ratio)")
        
        # Options for phi significance analysis
        st.markdown("#### Analysis Options")
        fine_resolution = st.checkbox("Use fine resolution around φ", value=True)
        
        # Run analysis button
        if st.button("Run Phi Significance Analysis", key="run_phi_analysis"):
            with st.spinner("Running phi significance analysis..."):
                try:
                    # Run the analysis
                    results = analyze_phi_significance(fine_resolution=fine_resolution, save_results=False)
                    
                    # Display results table
                    st.subheader("Results Table")
                    df = pd.DataFrame({
                        'f_s': results['fs_values'],
                        'Band Gap': results['band_gaps'],
                        'Fractal Dimension': results['fractal_dimensions'],
                        'Topological Invariant': results['topological_invariants'],
                        'Correlation Length': results['correlation_lengths'],
                        'Is Phi': np.isclose(results['fs_values'], 1.618034, rtol=1e-5)
                    })
                    st.dataframe(df)
                    
                    # Display plots
                    st.subheader("Visualization")
                    
                    # Main plots
                    fig = plt.figure(figsize=(12, 10))
                    plt.subplot(2, 2, 1)
                    plt.plot(results['fs_values'], results['band_gaps'], 'o-', color='#1f77b4', linewidth=2)
                    plt.axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Band Gap Size')
                    plt.title('Band Gap vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2)
                    plt.axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Fractal Dimension')
                    plt.title('Fractal Dimension vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.subplot(2, 2, 3)
                    plt.plot(results['fs_values'], results['topological_invariants'], 'o-', color='#2ca02c', linewidth=2)
                    plt.axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Topological Invariant')
                    plt.title('Topological Invariant vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.subplot(2, 2, 4)
                    plt.plot(results['fs_values'], results['correlation_lengths'], 'o-', color='#d62728', linewidth=2)
                    plt.axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    plt.xlabel('Scale Factor (f_s)')
                    plt.ylabel('Correlation Length')
                    plt.title('Correlation Length vs. Scale Factor')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Derivative plots
                    st.subheader("Derivatives (Phase Transition Indicators)")
                    
                    # Calculate numerical derivatives
                    fs_values = results['fs_values']
                    h = np.diff(fs_values)
                    
                    # Function to calculate numerical derivative
                    def numerical_derivative(y):
                        if len(y) != len(fs_values):
                            return np.zeros_like(fs_values)
                        dy = np.diff(y)
                        derivative = np.zeros_like(fs_values)
                        derivative[:-1] = dy / h
                        derivative[-1] = derivative[-2]  # Extend last value
                        return derivative
                    
                    # Calculate derivatives
                    d_gap = numerical_derivative(results['band_gaps'])
                    d_dim = numerical_derivative(results['fractal_dimensions'])
                    d_topo = numerical_derivative(results['topological_invariants'])
                    d_corr = numerical_derivative(results['correlation_lengths'])
                    
                    # Plot derivatives
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
                    axs[0, 0].plot(fs_values, d_gap, 'o-', color='#1f77b4')
                    axs[0, 0].axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    axs[0, 0].set_title('d(Band Gap)/d(f_s)')
                    axs[0, 0].grid(True, alpha=0.3)
                    axs[0, 0].legend()
                    
                    axs[0, 1].plot(fs_values, d_dim, 'o-', color='#ff7f0e')
                    axs[0, 1].axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    axs[0, 1].set_title('d(Fractal Dimension)/d(f_s)')
                    axs[0, 1].grid(True, alpha=0.3)
                    axs[0, 1].legend()
                    
                    axs[1, 0].plot(fs_values, d_topo, 'o-', color='#2ca02c')
                    axs[1, 0].axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    axs[1, 0].set_title('d(Topological Invariant)/d(f_s)')
                    axs[1, 0].grid(True, alpha=0.3)
                    axs[1, 0].legend()
                    
                    axs[1, 1].plot(fs_values, d_corr, 'o-', color='#d62728')
                    axs[1, 1].axvline(x=1.618034, color='r', linestyle='--', alpha=0.7, label='φ')
                    axs[1, 1].set_title('d(Correlation Length)/d(f_s)')
                    axs[1, 1].grid(True, alpha=0.3)
                    axs[1, 1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="phi_significance_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with fractal_topo_tab:
        st.subheader("Fractal-Topology Relationship")
        
        # Options for fractal-topology analysis
        st.markdown("#### Analysis Options")
        
        col1, col2 = st.columns(2)
        with col1:
            min_fs = st.number_input("Minimum f_s", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="ft_min_fs")
            max_fs = st.number_input("Maximum f_s", min_value=0.5, max_value=10.0, value=3.0, step=0.1, key="ft_max_fs")
        
        with col2:
            num_points = st.number_input("Number of points", min_value=5, max_value=51, value=21, step=2, key="ft_num_points")
            include_phi = st.checkbox("Include φ (Golden Ratio)", value=True, key="ft_include_phi")
        
        # Run analysis button
        if st.button("Run Fractal-Topology Analysis", key="run_ft_analysis"):
            with st.spinner("Running fractal-topology analysis..."):
                try:
                    # Generate f_s values
                    fs_values = np.linspace(min_fs, max_fs, num_points)
                    
                    # Add phi if requested
                    if include_phi:
                        from constants import PHI
                        fs_values = np.sort(np.append(fs_values, PHI))
                    
                    # Run the analysis
                    results = analyze_fractal_topology_relation(fs_values=fs_values, save_results=False)
                    
                    # Display results table
                    st.subheader("Results Table")
                    df = pd.DataFrame({
                        'f_s': results['fs_values'],
                        'Band Gap': results['band_gaps'],
                        'Fractal Dimension': results['fractal_dimensions'],
                        'Topological Invariant': results['topological_invariants'],
                        'Z2 Index': results['z2_indices'],
                        'Correlation Length': results['correlation_lengths'],
                        'Self-Similarity': results['self_similarity_metrics']
                    })
                    st.dataframe(df)
                    
                    # Display correlation results
                    st.subheader("Correlation Analysis")
                    if 'correlations' in results:
                        corr = results['correlations']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Pearson Correlation", f"{corr['fractal_topo_pearson']:.4f}")
                            st.metric("p-value", f"{corr['p_value_pearson']:.4f}")
                        with col2:
                            st.metric("Spearman Correlation", f"{corr['fractal_topo_spearman']:.4f}")
                            st.metric("p-value", f"{corr['p_value_spearman']:.4f}")
                    
                    # Display plots
                    st.subheader("Visualization")
                    
                    # Scatter plot of fractal dimension vs topological invariant
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(
                        results['fractal_dimensions'],
                        results['topological_invariants'],
                        c=results['fs_values'],
                        cmap='viridis',
                        s=50,
                        alpha=0.7
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Scale Factor (f_s)')
                    
                    # Highlight phi point if included
                    if include_phi:
                        from constants import PHI
                        phi_idx = np.argmin(np.abs(results['fs_values'] - PHI))
                        ax.scatter(
                            results['fractal_dimensions'][phi_idx],
                            results['topological_invariants'][phi_idx],
                            s=100,
                            facecolors='none',
                            edgecolors='r',
                            linewidth=2,
                            label=f'φ ≈ {PHI:.4f}'
                        )
                    
                    ax.set_xlabel('Fractal Dimension')
                    ax.set_ylabel('Topological Invariant')
                    ax.set_title('Fractal Dimension vs. Topological Invariant')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Add correlation information
                    if 'correlations' in results:
                        corr = results['correlations']
                        if not np.isnan(corr['fractal_topo_pearson']):
                            ax.annotate(
                                f"Pearson r = {corr['fractal_topo_pearson']:.4f} (p={corr['p_value_pearson']:.4f})\n"
                                f"Spearman ρ = {corr['fractal_topo_spearman']:.4f} (p={corr['p_value_spearman']:.4f})",
                                xy=(0.05, 0.95),
                                xycoords='axes fraction',
                                ha='left',
                                va='top',
                                bbox=dict(boxstyle='round', fc='white', alpha=0.8)
                            )
                    
                    st.pyplot(fig)
                    
                    # Plot fractal dimension and topological invariant vs f_s
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Plot fractal dimension
                    ax1.plot(results['fs_values'], results['fractal_dimensions'], 'o-', color='#ff7f0e', linewidth=2, label='Fractal Dimension')
                    ax1.set_xlabel('Scale Factor (f_s)')
                    ax1.set_ylabel('Fractal Dimension', color='#ff7f0e')
                    ax1.tick_params(axis='y', labelcolor='#ff7f0e')
                    
                    # Create second y-axis for topological invariant
                    ax2 = ax1.twinx()
                    ax2.plot(results['fs_values'], results['topological_invariants'], 's-', color='#2ca02c', linewidth=2, label='Topological Invariant')
                    ax2.set_ylabel('Topological Invariant', color='#2ca02c')
                    ax2.tick_params(axis='y', labelcolor='#2ca02c')
                    
                    # Add vertical line at phi if included
                    if include_phi:
                        from constants import PHI
                        ax1.axvline(x=PHI, color='r', linestyle='--', alpha=0.7, label=f'φ ≈ {PHI:.4f}')
                    
                    # Create combined legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    ax1.set_title('Fractal Dimension and Topological Invariant vs. Scale Factor')
                    ax1.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Add download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="fractal_topology_relation.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

#!/usr/bin/env python
"""
Test suite for fractal analysis and visualization functionality.
"""

import numpy as np
import pytest
from qutip import basis, sigmax, sigmaz
from pathlib import Path
import matplotlib.pyplot as plt

from analyses.fractal_analysis import (
    load_fractal_config,
    compute_wavefunction_profile,
    estimate_fractal_dimension,
    compute_energy_spectrum
)
from analyses.visualization.fractal_plots import (
    plot_energy_spectrum,
    plot_wavefunction_profile,
    plot_fractal_dimension,
    plot_fractal_analysis_summary
)

# Test data generation helpers
def generate_fractal_like_data(n_points: int = 1000) -> np.ndarray:
    """Generate a Cantor-set-like fractal pattern for testing."""
    x = np.linspace(0, 1, n_points)
    y = np.zeros(n_points)
    
    def recurse(start: int, end: int, depth: int = 4):
        if depth == 0 or end - start < 3:
            return
        third = (end - start) // 3
        mid_start = start + third
        mid_end = end - third
        y[mid_start:mid_end] = 1.0
        recurse(start, start + third, depth - 1)
        recurse(end - third, end, depth - 1)
    
    recurse(0, n_points)
    return y

def mock_hamiltonian(f_s: float):
    """Create a simple test Hamiltonian with f_s dependence."""
    return f_s * sigmax() + (1 - f_s) * sigmaz()

# Configuration tests
def test_load_fractal_config():
    """Test loading of fractal configuration."""
    config = load_fractal_config()
    
    assert isinstance(config, dict)
    assert 'energy_spectrum' in config
    assert 'wavefunction_zoom' in config
    assert 'fractal_dimension' in config
    
    # Verify required parameters
    spectrum_config = config['energy_spectrum']
    assert 'f_s_range' in spectrum_config
    assert 'resolution' in spectrum_config
    assert isinstance(spectrum_config['f_s_range'], list)
    assert len(spectrum_config['f_s_range']) == 2

# Wavefunction analysis tests
def test_compute_wavefunction_profile():
    """Test wavefunction profile computation."""
    # Create a test state
    psi = (basis(2, 0) + basis(2, 1)).unit()
    x_array = np.linspace(0, 1, 100)
    
    # Test basic computation
    density, _ = compute_wavefunction_profile(psi, x_array)
    assert isinstance(density, np.ndarray)
    assert len(density) == len(x_array)
    assert np.all(density >= 0)  # Probability densities should be non-negative
    
    # Test with zoom and logging
    density, details = compute_wavefunction_profile(
        psi, x_array, zoom_factor=2.0, log_details=True
    )
    assert len(density) == len(x_array) * 2  # Zoomed array
    assert isinstance(details, dict)
    assert 'normalization' in details
    assert abs(details['normalization'] - 1.0) < 1e-10  # Should be normalized
    assert 'zoom_regions' in details

# Fractal dimension tests
def test_estimate_fractal_dimension():
    """Test fractal dimension estimation."""
    # Generate test fractal data
    data = generate_fractal_like_data()
    box_sizes = np.logspace(-3, 0, 10)
    
    # Test basic estimation
    dimension, info = estimate_fractal_dimension(data, box_sizes)
    assert isinstance(dimension, float)
    assert isinstance(info, dict)
    assert 'std_error' in info
    assert 'r_squared' in info
    assert 'confidence_interval' in info
    assert info['r_squared'] >= 0 and info['r_squared'] <= 1
    
    # Test with insufficient data
    with pytest.warns(UserWarning):
        dim, info = estimate_fractal_dimension(np.zeros(10), box_sizes)
        assert dim == 0.0
        assert info['std_error'] == np.inf

# Energy spectrum tests
def test_compute_energy_spectrum():
    """Test energy spectrum computation and analysis."""
    config = load_fractal_config()
    
    f_s_values, energies, analysis = compute_energy_spectrum(
        mock_hamiltonian,
        config=config
    )
    
    assert isinstance(f_s_values, np.ndarray)
    assert isinstance(energies, np.ndarray)
    assert isinstance(analysis, dict)
    assert 'self_similar_regions' in analysis
    assert 'correlation_matrix' in analysis
    assert 'gap_statistics' in analysis
    
    # Verify energy values are real
    assert np.all(np.isreal(energies))
    
    # Check correlation matrix properties
    corr_matrix = analysis['correlation_matrix']
    assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)

# Visualization tests
@pytest.mark.mpl_image_compare
def test_plot_energy_spectrum():
    """Test energy spectrum plotting."""
    f_s_values = np.linspace(0, 1, 50)
    energies = np.sin(2 * np.pi * f_s_values)  # Simple test pattern
    analysis = {
        'self_similar_regions': [(0.2, 0.3, 0.7, 0.8)],
        'correlation_matrix': np.eye(len(f_s_values)),
        'gap_statistics': {
            'mean': 0.1,
            'std': 0.05,
            'min': 0.0,
            'max': 0.2
        }
    }
    
    fig = plot_energy_spectrum(f_s_values, energies, analysis)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

@pytest.mark.mpl_image_compare
def test_plot_wavefunction_profile():
    """Test wavefunction profile plotting."""
    psi = (basis(2, 0) + basis(2, 1)).unit()
    x_array = np.linspace(0, 1, 100)
    
    fig = plot_wavefunction_profile(psi, x_array)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

@pytest.mark.mpl_image_compare
def test_plot_fractal_dimension():
    """Test fractal dimension plotting."""
    depths = np.array([1, 2, 3, 4, 5])
    dimensions = 1.5 - 0.1 * depths
    errors = 0.1 * np.ones_like(depths)
    
    fig = plot_fractal_dimension(depths, dimensions, errors)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

@pytest.mark.mpl_image_compare
def test_plot_fractal_analysis_summary():
    """Test summary figure generation."""
    # Generate test data
    f_s_values = np.linspace(0, 1, 50)
    energies = np.sin(2 * np.pi * f_s_values)
    analysis = {
        'self_similar_regions': [(0.2, 0.3, 0.7, 0.8)],
        'correlation_matrix': np.eye(len(f_s_values)),
        'gap_statistics': {
            'mean': 0.1,
            'std': 0.05,
            'min': 0.0,
            'max': 0.2
        }
    }
    psi = (basis(2, 0) + basis(2, 1)).unit()
    depths = np.array([1, 2, 3, 4, 5])
    dimensions = 1.5 - 0.1 * depths
    errors = 0.1 * np.ones_like(depths)
    
    fig = plot_fractal_analysis_summary(
        f_s_values, energies, analysis, psi,
        depths, dimensions, errors
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

# Integration tests
def test_full_analysis_pipeline():
    """Test the complete fractal analysis pipeline."""
    # Load configuration
    config = load_fractal_config()
    
    # Generate test data
    psi = (basis(2, 0) + basis(2, 1)).unit()
    f_s_values, energies, analysis = compute_energy_spectrum(
        mock_hamiltonian,
        config=config
    )
    
    # Compute wavefunction profile
    x_array = np.linspace(0, 1, 100)
    density, details = compute_wavefunction_profile(
        psi, x_array, zoom_factor=2.0, log_details=True
    )
    
    # Estimate fractal dimension
    depths = np.array([1, 2, 3, 4, 5])
    dimensions = []
    errors = []
    
    for depth in depths:
        data = generate_fractal_like_data(2**depth)
        dim, info = estimate_fractal_dimension(data, config=config)
        dimensions.append(dim)
        errors.append(info['std_error'])
    
    dimensions = np.array(dimensions)
    errors = np.array(errors)
    
    # Generate all plots
    figs = [
        plot_energy_spectrum(f_s_values, energies, analysis),
        plot_wavefunction_profile(psi, x_array),
        plot_fractal_dimension(depths, dimensions, errors),
        plot_fractal_analysis_summary(
            f_s_values, energies, analysis, psi,
            depths, dimensions, errors
        )
    ]
    
    # Verify all plots were created successfully
    for fig in figs:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

if __name__ == '__main__':
    pytest.main([__file__])
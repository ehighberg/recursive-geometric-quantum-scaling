# -*- coding: utf-8 -*-
"""
Tests for the scaling analysis functionality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from qutip import Qobj, sigmaz, sigmax, tensor, qeye

from analyses.scaling import analyze_fs_scaling, analyze_phi_significance, analyze_fractal_topology_relation
from app.scaling_analysis import display_scaling_analysis

class MockResult:
    """Mock simulation result for testing."""
    def __init__(self):
        # Create a simple Hamiltonian function
        def hamiltonian(f_s):
            return f_s * (tensor(sigmaz(), qeye(2)) + 0.5 * tensor(qeye(2), sigmax()))
        
        self.hamiltonian = hamiltonian
        self.states = [Qobj(np.eye(4)) for _ in range(10)]  # Mock states
        self.times = np.linspace(0, 10, 10)  # Mock times

def test_analyze_fs_scaling():
    """Test the analyze_fs_scaling function."""
    # Define test f_s values
    fs_values = [0.5, 1.0, 1.5, 2.0]
    
    # Run analysis
    results = analyze_fs_scaling(fs_values=fs_values, save_results=False)
    
    # Check that results contain expected keys
    assert 'fs_values' in results
    assert 'band_gaps' in results
    assert 'fractal_dimensions' in results
    assert 'topological_invariants' in results
    assert 'correlation_lengths' in results
    
    # Check that arrays have correct length
    assert len(results['fs_values']) == len(fs_values)
    assert len(results['band_gaps']) == len(fs_values)
    assert len(results['fractal_dimensions']) == len(fs_values)
    assert len(results['topological_invariants']) == len(fs_values)
    assert len(results['correlation_lengths']) == len(fs_values)

def test_analyze_phi_significance():
    """Test the analyze_phi_significance function."""
    # Run analysis with minimal resolution for faster testing
    results = analyze_phi_significance(fine_resolution=False, save_results=False)
    
    # Check that results contain expected keys
    assert 'fs_values' in results
    assert 'band_gaps' in results
    assert 'fractal_dimensions' in results
    assert 'topological_invariants' in results
    assert 'correlation_lengths' in results
    assert 'energy_spectra' in results
    assert 'wavefunction_profiles' in results
    
    # Check that arrays have correct length
    n_values = len(results['fs_values'])
    assert len(results['band_gaps']) == n_values
    assert len(results['fractal_dimensions']) == n_values
    assert len(results['topological_invariants']) == n_values
    assert len(results['correlation_lengths']) == n_values
    assert len(results['energy_spectra']) == n_values

def test_analyze_fractal_topology_relation():
    """Test the analyze_fractal_topology_relation function."""
    # Define test f_s values
    fs_values = np.linspace(0.5, 3.0, 6)
    
    # Run analysis
    results = analyze_fractal_topology_relation(fs_values=fs_values, save_results=False)
    
    # Check that results contain expected keys
    assert 'fs_values' in results
    assert 'band_gaps' in results
    assert 'fractal_dimensions' in results
    assert 'topological_invariants' in results
    assert 'correlation_lengths' in results
    assert 'z2_indices' in results
    assert 'energy_spectra' in results
    assert 'self_similarity_metrics' in results
    assert 'correlations' in results
    
    # Check that arrays have correct length
    assert len(results['fs_values']) == len(fs_values)
    assert len(results['band_gaps']) == len(fs_values)
    assert len(results['fractal_dimensions']) == len(fs_values)
    assert len(results['topological_invariants']) == len(fs_values)
    assert len(results['correlation_lengths']) == len(fs_values)
    assert len(results['z2_indices']) == len(fs_values)
    assert len(results['self_similarity_metrics']) == len(fs_values)
    
    # Check correlation results
    assert 'fractal_topo_pearson' in results['correlations']
    assert 'p_value_pearson' in results['correlations']
    assert 'fractal_topo_spearman' in results['correlations']
    assert 'p_value_spearman' in results['correlations']

@pytest.mark.parametrize("mode", [
    "Pulse Sequence Evolution",
    "Amplitude-Scaled Evolution",
    "Quantum Gate Operations",
    "Topological Braiding"
])
def test_display_scaling_analysis(mode):
    """Test the display_scaling_analysis function with different modes."""
    # Create mock result
    result = MockResult()
    
    # Mock streamlit components
    with patch('app.scaling_analysis.st') as mock_st:
        # Set up mock tabs
        mock_tabs = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = mock_tabs
        
        # Call the function
        display_scaling_analysis(result, mode)
        
        # Verify that tabs were created
        mock_st.tabs.assert_called_once()
        
        # Verify that warning is not shown (since result has hamiltonian)
        mock_st.warning.assert_not_called()

def test_display_scaling_analysis_no_hamiltonian():
    """Test the display_scaling_analysis function with a result that has no hamiltonian."""
    # Create mock result without hamiltonian
    result = MagicMock()
    delattr(result, 'hamiltonian')
    
    # Mock streamlit components
    with patch('app.scaling_analysis.st') as mock_st:
        # Call the function
        display_scaling_analysis(result)
        
        # Verify that warning is shown
        mock_st.warning.assert_called_once()

def test_display_scaling_analysis_no_result():
    """Test the display_scaling_analysis function with no result."""
    # Mock streamlit components
    with patch('app.scaling_analysis.st') as mock_st:
        # Call the function with None
        display_scaling_analysis(None)
        
        # Verify that warning is shown
        mock_st.warning.assert_called_once()

def test_integration_with_app():
    """Test the integration of scaling analysis with the app."""
    # Create mock result
    result = MockResult()
    
    # Mock streamlit components
    with patch('app.scaling_analysis.st') as mock_st, \
         patch('app.scaling_analysis.analyze_fs_scaling') as mock_analyze_fs, \
         patch('app.scaling_analysis.analyze_phi_significance') as mock_analyze_phi, \
         patch('app.scaling_analysis.analyze_fractal_topology_relation') as mock_analyze_ft:
        
        # Set up mock tabs and buttons
        mock_tabs = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = mock_tabs
        
        # Set up mock button returns
        mock_st.button.side_effect = [True, True, True]  # All buttons return True
        
        # Set up mock number_input returns
        mock_st.number_input.side_effect = [0.5, 3.0, 6, 0.5, 3.0, 21]
        
        # Set up mock checkbox returns
        mock_st.checkbox.side_effect = [True, False, True]
        
        # Set up mock analysis returns
        mock_analyze_fs.return_value = {
            'fs_values': np.array([0.5, 1.0, 1.5, 2.0]),
            'band_gaps': np.array([0.1, 0.2, 0.3, 0.4]),
            'fractal_dimensions': np.array([1.1, 1.2, 1.3, 1.4]),
            'topological_invariants': np.array([0, 1, 0, 1]),
            'correlation_lengths': np.array([10.0, 5.0, 3.3, 2.5])
        }
        
        mock_analyze_phi.return_value = {
            'fs_values': np.array([0.5, 1.0, 1.5, 2.0]),
            'band_gaps': np.array([0.1, 0.2, 0.3, 0.4]),
            'fractal_dimensions': np.array([1.1, 1.2, 1.3, 1.4]),
            'topological_invariants': np.array([0, 1, 0, 1]),
            'correlation_lengths': np.array([10.0, 5.0, 3.3, 2.5]),
            'energy_spectra': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]), np.array([0.7, 0.8])],
            'wavefunction_profiles': [None, None, None, None]
        }
        
        mock_analyze_ft.return_value = {
            'fs_values': np.array([0.5, 1.0, 1.5, 2.0]),
            'band_gaps': np.array([0.1, 0.2, 0.3, 0.4]),
            'fractal_dimensions': np.array([1.1, 1.2, 1.3, 1.4]),
            'topological_invariants': np.array([0, 1, 0, 1]),
            'correlation_lengths': np.array([10.0, 5.0, 3.3, 2.5]),
            'z2_indices': np.array([0, 1, 0, 1]),
            'energy_spectra': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]), np.array([0.7, 0.8])],
            'self_similarity_metrics': np.array([1, 2, 3, 4]),
            'correlations': {
                'fractal_topo_pearson': 0.8,
                'p_value_pearson': 0.01,
                'fractal_topo_spearman': 0.7,
                'p_value_spearman': 0.02
            }
        }
        
        # Call the function
        display_scaling_analysis(result, "Amplitude-Scaled Evolution")
        
        # Verify that tabs were created
        mock_st.tabs.assert_called_once()
        
        # Verify that analysis functions were called
        mock_analyze_fs.assert_called_once()
        mock_analyze_phi.assert_called_once()
        mock_analyze_ft.assert_called_once()
        
        # Verify that plots were created
        assert mock_st.pyplot.call_count >= 6  # At least 6 plots should be created

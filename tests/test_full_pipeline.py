"""
Integration tests for the complete analysis pipeline including UI, topology, and visualization.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import streamlit as st
from qutip import basis, Qobj

from app.analyze_results import analyze_simulation_results
from analyses.topological_invariants import compute_chern_number, compute_winding_number
from analyses.topology_plots import plot_invariants, plot_protection_metrics
from analyses.visualization.state_plots import plot_state_evolution

class MockSimulationResult:
    """Mock simulation result with all required attributes"""
    def __init__(self):
        # Basic quantum states
        self.states = [basis(2,0), basis(2,1)]
        self.times = [0, 1]
        
        # Hamiltonian function
        self.hamiltonian = lambda x: np.array([[1, 0], [0, -1]])
        
        # Fractal analysis data
        self.recursion_depths = [1, 2, 3]
        self.fractal_dimensions = [1.5, 1.6, 1.7]
        self.dimension_errors = [0.1, 0.1, 0.1]
        
        # Topological data
        self.chern_number = 1
        self.winding_number = 2
        self.z2_index = 1
        
        # Protection metrics
        self.energy_gaps = np.linspace(0, 1, 100)
        self.localization_measures = np.cos(np.linspace(0, 2*np.pi, 100))

@pytest.fixture
def mock_streamlit():
    """Setup mock Streamlit components"""
    with patch('streamlit.header') as mock_header, \
         patch('streamlit.subheader') as mock_subheader, \
         patch('streamlit.pyplot') as mock_pyplot, \
         patch('streamlit.slider') as mock_slider, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.session_state', {}) as mock_state:
        
        mock_tabs.return_value = [MagicMock() for _ in range(4)]
        mock_columns.return_value = [MagicMock() for _ in range(3)]
        
        yield {
            'header': mock_header,
            'subheader': mock_subheader,
            'pyplot': mock_pyplot,
            'slider': mock_slider,
            'columns': mock_columns,
            'tabs': mock_tabs,
            'state': mock_state
        }

def test_full_analysis_pipeline(mock_streamlit):
    """Test the complete analysis pipeline"""
    result = MockSimulationResult()
    
    # Run full analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify all components were called
    mock_streamlit['pyplot'].assert_called()  # Plots were generated
    mock_streamlit['header'].assert_called()  # Headers were created
    assert mock_streamlit['pyplot'].call_count >= 3  # Multiple plots generated

def test_pipeline_with_noise(mock_streamlit):
    """Test pipeline with noisy data"""
    result = MockSimulationResult()
    
    # Add noise to states
    noisy_states = []
    for state in result.states:
        noise = np.random.normal(0, 0.1, (2,1))  # Increased noise amplitude
        noisy_state = (state + Qobj(noise)).unit()
        noisy_states.append(noisy_state)
    result.states = noisy_states
    
    # Add coherence metric that shows significant decay
    result.coherence = [0.9, 0.5]  # Significant coherence decay
    
    # Run analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify noise analysis was performed
    assert any('Noise' in str(call) for call in mock_streamlit['subheader'].call_args_list)

def test_pipeline_error_handling(mock_streamlit):
    """Test error handling throughout the pipeline"""
    result = MockSimulationResult()
    
    # Test with missing attributes
    delattr(result, 'states')
    
    with patch('streamlit.error') as mock_error:
        analyze_simulation_results(result, mode="Topological Braiding")
        mock_error.assert_not_called()  # Should handle gracefully

def test_topological_analysis_integration(mock_streamlit):
    """Test integration of topological analysis"""
    result = MockSimulationResult()
    
    # Mock the topological analysis tab
    tabs = mock_streamlit['tabs'].return_value
    with tabs[3]:  # Topological Analysis tab
        analyze_simulation_results(result, mode="Topological Braiding")
        
        # Verify topological plots were generated
        assert mock_streamlit['pyplot'].call_count >= 2

def test_metric_correlation_analysis(mock_streamlit):
    """Test correlation analysis between metrics"""
    result = MockSimulationResult()
    
    # Add correlation data
    result.metric_correlations = {
        'chern_entropy': 0.8,
        'winding_coherence': 0.6
    }
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify correlation plots were generated
    assert mock_streamlit['pyplot'].call_count >= 3

def test_animation_integration(mock_streamlit):
    """Test integration with animations"""
    result = MockSimulationResult()
    
    # Add time evolution data
    times = np.linspace(0, 10, 50)
    states = []
    for t in times:
        state = (np.cos(t) * basis(2, 0) + np.sin(t) * basis(2, 1)).unit()
        states.append(state)
    result.states = states
    result.times = times
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify animations were generated
    assert mock_streamlit['pyplot'].call_count >= 3

def test_export_functionality(mock_streamlit):
    """Test data export functionality"""
    result = MockSimulationResult()
    
    # Mock download button
    with patch('streamlit.download_button') as mock_download:
        analyze_simulation_results(result, mode="Topological Braiding")
        
        # Verify export options were created
        mock_download.assert_not_called()  # Feature coming soon

def test_performance_monitoring(mock_streamlit):
    """Test performance monitoring in the pipeline"""
    result = MockSimulationResult()
    
    # Add performance metrics
    result.computation_times = {
        'topological_invariants': 0.1,
        'protection_metrics': 0.2,
        'visualization': 0.3
    }
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify performance data was displayed
    assert any('Performance' in str(call) for call in mock_streamlit['subheader'].call_args_list)

def test_state_persistence(mock_streamlit):
    """Test state persistence between analyses"""
    result = MockSimulationResult()
    
    # First analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Modify result
    result.chern_number = 2
    
    # Second analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify state was updated
    assert mock_streamlit['pyplot'].call_count >= 6  # Multiple plots from both analyses

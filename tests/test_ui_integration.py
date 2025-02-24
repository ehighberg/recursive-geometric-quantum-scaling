"""
Tests for UI integration and Streamlit components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import streamlit as st
from app.analyze_results import analyze_simulation_results, display_experiment_summary
from analyses.topology_plots import plot_invariants, plot_protection_metrics

class MockResult:
    """Mock simulation result object for testing"""
    def __init__(self, with_states=True, with_hamiltonian=True):
        from qutip import basis
        if with_states:
            self.states = [basis(2, 0), basis(2, 1)]
            self.times = [0, 1]
        if with_hamiltonian:
            self.hamiltonian = lambda x: np.array([[1, 0], [0, -1]])
        self.recursion_depths = [1, 2, 3]
        self.fractal_dimensions = [1.5, 1.6, 1.7]
        self.dimension_errors = [0.1, 0.1, 0.1]

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components"""
    with patch('streamlit.header') as mock_header, \
         patch('streamlit.subheader') as mock_subheader, \
         patch('streamlit.pyplot') as mock_pyplot, \
         patch('streamlit.slider') as mock_slider, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.tabs') as mock_tabs:
        
        mock_header.return_value = None
        mock_subheader.return_value = None
        mock_pyplot.return_value = None
        mock_slider.return_value = (0.0, 5.0)
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_tabs.return_value = [MagicMock() for _ in range(6)]
        
        yield {
            'header': mock_header,
            'subheader': mock_subheader,
            'pyplot': mock_pyplot,
            'slider': mock_slider,
            'columns': mock_columns,
            'tabs': mock_tabs
        }

def test_topological_analysis_tab(mock_streamlit):
    """Test the topological analysis tab functionality"""
    result = MockResult()
    
    # Mock the tab context
    with patch('streamlit.tabs') as mock_tabs:
        tabs = [MagicMock() for _ in range(6)]
        mock_tabs.return_value = tabs
        
        # Test that topological analysis components are created
        with tabs[4]:  # Topological Analysis tab
            st.header("Topological Analysis")
            control_range = st.slider("Topological Control Parameter Range", 0.0, 10.0, (0.0, 5.0))
            
            # Verify slider was created with correct parameters
            mock_streamlit['slider'].assert_called_with(
                "Topological Control Parameter Range", 0.0, 10.0, (0.0, 5.0)
            )
            
            # Verify plots were created
            assert mock_streamlit['pyplot'].call_count >= 2  # Should have at least 2 plots

def test_analyze_results_with_topology(mock_streamlit):
    """Test analyze_results function with topological data"""
    result = MockResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify that analysis components were created
    mock_streamlit['header'].assert_called()
    mock_streamlit['pyplot'].assert_called()

def test_display_experiment_summary_with_topology(mock_streamlit):
    """Test experiment summary display with topological data"""
    result = MockResult()
    
    display_experiment_summary(result)
    
    # Verify summary components were created
    mock_streamlit['header'].assert_called_with("Experiment Summary")
    mock_streamlit['subheader'].assert_called()

def test_ui_error_handling(mock_streamlit):
    """Test UI error handling for invalid inputs"""
    result = MockResult(with_states=False, with_hamiltonian=False)
    
    # Mock error display
    with patch('streamlit.error') as mock_error:
        analyze_simulation_results(result, mode="Invalid Mode")
        mock_error.assert_not_called()  # Should handle invalid mode gracefully

def test_topological_metrics_display(mock_streamlit):
    """Test display of topological metrics"""
    result = MockResult()
    
    # Add mock topological metrics
    result.chern_number = 1
    result.winding_number = 2
    result.z2_index = 1
    
    with patch('streamlit.metric') as mock_metric:
        display_experiment_summary(result)
        # Verify metrics were displayed
        assert mock_metric.call_count >= 3

def test_ui_state_persistence(mock_streamlit):
    """Test UI state persistence between reruns"""
    # Mock session state
    with patch('streamlit.session_state', {}) as mock_state:
        result = MockResult()
        mock_state['simulation_results'] = result
        
        analyze_simulation_results(result, mode="Topological Braiding")
        # Verify state was preserved
        assert 'simulation_results' in mock_state

def test_responsive_layout(mock_streamlit):
    """Test responsive layout behavior"""
    result = MockResult()
    
    # Test with different column configurations
    with patch('streamlit.columns') as mock_columns:
        # Test 3-column layout
        mock_columns.return_value = [MagicMock() for _ in range(3)]
        display_experiment_summary(result)
        mock_columns.assert_called()
        
        # Test 2-column layout
        mock_columns.return_value = [MagicMock() for _ in range(2)]
        display_experiment_summary(result)
        mock_columns.assert_called()

def test_plot_updates(mock_streamlit):
    """Test plot updates based on user interaction"""
    result = MockResult()
    
    # Mock slider interaction
    mock_streamlit['slider'].return_value = (1.0, 6.0)
    
    with patch('analyses.topology_plots.plot_invariants') as mock_plot:
        analyze_simulation_results(result, mode="Topological Braiding")
        # Verify plot was updated with new range
        mock_plot.assert_called()

def test_export_functionality(mock_streamlit):
    """Test export functionality in Raw Data tab"""
    result = MockResult()
    
    # Mock button and download components
    with patch('streamlit.button') as mock_button, \
         patch('streamlit.download_button') as mock_download:
        mock_button.return_value = True
        
        display_experiment_summary(result)
        
        # Verify export options were created
        mock_button.assert_called()
        # Download functionality is coming soon, so download_button shouldn't be called
        mock_download.assert_not_called()
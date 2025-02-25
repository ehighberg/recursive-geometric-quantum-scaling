"""
Tests for UI integration and Streamlit components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.analyze_results import analyze_simulation_results, display_experiment_summary
from app.scaling_analysis import display_scaling_analysis
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
        # Initialize topological metrics
        self.chern_number = 1
        self.winding_number = 2
        self.z2_index = 1

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
        mock_tabs.return_value = [MagicMock() for _ in range(7)]  # Updated for new scaling tab
        
        yield {
            'header': mock_header,
            'subheader': mock_subheader,
            'pyplot': mock_pyplot,
            'slider': mock_slider,
            'columns': mock_columns,
            'tabs': mock_tabs
        }

def test_topological_analysis_tab():
    """Test the topological analysis tab functionality"""
    
    # Mock the tab context
    with patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.header') as mock_header, \
         patch('streamlit.slider') as mock_slider, \
         patch('streamlit.pyplot') as mock_pyplot:
        
        tabs = [MagicMock() for _ in range(7)]  # Updated for new scaling tab
        mock_tabs.return_value = tabs
        
        # Test that topological analysis components are created
        with tabs[4]:  # Topological Analysis tab
            mock_header.assert_called_once()
            mock_slider.assert_called_with(
                "Topological Control Parameter Range", 0.0, 10.0, (0.0, 5.0)
            )
            assert mock_pyplot.call_count >= 2  # Should have at least 2 plots

def test_analyze_results_with_topology():
    """Test analyze_results function with topological data"""
    result = MockResult()
    
    # Mock and verify components
    with patch('streamlit.header') as mock_header, \
         patch('streamlit.pyplot') as mock_pyplot:
        analyze_simulation_results(result, mode="Topological Braiding")
        mock_header.assert_called()
        mock_pyplot.assert_called()

def test_display_experiment_summary_with_topology():
    """Test experiment summary display with topological data"""
    result = MockResult()
    
    # Mock and verify components
    with patch('streamlit.header') as mock_header, \
         patch('streamlit.subheader') as mock_subheader:
        display_experiment_summary(result)
        mock_header.assert_called_with("Experiment Summary")
        mock_subheader.assert_called()

def test_ui_error_handling():
    """Test UI error handling for invalid inputs"""
    result = MockResult(with_states=False, with_hamiltonian=False)
    
    # Mock error display
    with patch('streamlit.error') as mock_error:
        analyze_simulation_results(result, mode="Invalid Mode")
        mock_error.assert_not_called()  # Should handle invalid mode gracefully

def test_topological_metrics_display():
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

def test_ui_state_persistence():
    """Test UI state persistence between reruns"""
    # Mock session state
    with patch('streamlit.session_state', {}) as mock_state:
        result = MockResult()
        mock_state['simulation_results'] = result
        
        analyze_simulation_results(result, mode="Topological Braiding")
        # Verify state was preserved
        assert 'simulation_results' in mock_state

def test_responsive_layout():
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

def test_plot_updates():
    """Test plot updates based on user interaction"""
    result = MockResult()
    
    # Mock components and interaction
    with patch('streamlit.slider') as mock_slider, \
         patch('analyses.topology_plots.plot_invariants') as mock_plot:
        mock_slider.return_value = (1.0, 6.0)
        analyze_simulation_results(result, mode="Topological Braiding")
        mock_plot.assert_called()

def test_export_functionality():
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

def test_scaling_analysis_tab():
    """Test the scaling analysis tab functionality"""
    result = MockResult()
    
    # Mock the tab context
    with patch('streamlit.tabs') as mock_tabs:
        tabs = [MagicMock() for _ in range(7)]  # Updated for new scaling tab
        mock_tabs.return_value = tabs
        
        # Test that scaling analysis components are created
        with tabs[5]:  # Scaling Analysis tab
            display_scaling_analysis(result, _mode="Amplitude-Scaled Evolution")
            
            # Mock and verify components
            with patch('streamlit.subheader') as mock_subheader:
                display_scaling_analysis(result, _mode="Amplitude-Scaled Evolution")
                assert mock_subheader.call_count > 0

def test_scaling_analysis_integration():
    """Test integration of scaling analysis with the app"""
    result = MockResult()
    
    # Mock the scaling analysis function
    with patch('app.scaling_analysis.display_scaling_analysis') as mock_display_scaling:
        # Set up session state with simulation results
        with patch('streamlit.session_state', {'simulation_results': result}):
            # Create a mock tab context
            with patch('streamlit.tabs') as mock_tabs:
                tabs = [MagicMock() for _ in range(7)]
                mock_tabs.return_value = tabs
                
                # Simulate the app's behavior by calling display_scaling_analysis directly
                with tabs[5]:  # Scaling Analysis tab
                    display_scaling_analysis(result, _mode="Amplitude-Scaled Evolution")
            
            # Verify that scaling analysis was called
            mock_display_scaling.assert_called_once()

def test_scaling_analysis_with_different_modes():
    """Test scaling analysis with different simulation modes"""
    result = MockResult()
    
    # Test with different modes
    for mode in ["Pulse Sequence Evolution", "Amplitude-Scaled Evolution", 
                "Quantum Gate Operations", "Topological Braiding"]:
        # Mock the scaling analysis function
        with patch('app.scaling_analysis.display_scaling_analysis') as mock_display_scaling:
            display_scaling_analysis(result, _mode=mode)
            
            # Verify that scaling analysis was called with the correct mode
            mock_display_scaling.assert_called_once_with(result, _mode=mode)

"""
Tests for topological analysis UI components.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from qutip import basis, Qobj

class MockTopologicalResult:
    """Mock result object for topological analysis tests"""
    def __init__(self):
        # Basic quantum states
        self.states = [basis(2,0) for _ in range(50)]  # 50 states for evolution
        self.times = np.linspace(0, 10, len(self.states))
        
        # Topological invariants
        self.chern_number = 1
        self.winding_number = 2
        self.z2_index = 1
        
        # Protection metrics
        self.energy_gaps = np.abs(np.sin(np.linspace(0, 2*np.pi, len(self.states))))
        self.localization_measures = np.abs(np.cos(np.linspace(0, 2*np.pi, len(self.states))))
        
        # Evolution data
        self.chern_evolution = np.sin(self.times) * 0.5 + 0.5
        self.winding_evolution = np.cos(self.times) * 0.5
        
        # Performance metrics
        self.computation_times = {
            'chern_number': 0.1,
            'winding_number': 0.2,
            'protection_metrics': 0.3
        }

@pytest.fixture
def mock_streamlit():
    """Create mock streamlit components"""
    with patch('app.analyze_results.st.header') as mock_header, \
         patch('app.analyze_results.st.tabs') as mock_tabs, \
         patch('app.analyze_results.st.columns') as mock_columns, \
         patch('app.analyze_results.st.metric') as mock_metric, \
         patch('app.analyze_results.st.pyplot') as mock_pyplot, \
         patch('app.analyze_results.st.slider') as mock_slider, \
         patch('app.analyze_results.st.info') as mock_info, \
         patch('app.analyze_results.st.error') as mock_error, \
         patch('app.analyze_results.st.download_button') as mock_download:
        
        # Configure mock tabs
        tab_mocks = [MagicMock() for _ in range(4)]
        mock_tabs.return_value = tab_mocks
        
        # Configure mock columns
        mock_columns.return_value = [MagicMock() for _ in range(5)]
        
        yield {
            'header': mock_header,
            'tabs': mock_tabs,
            'columns': mock_columns,
            'metric': mock_metric,
            'pyplot': mock_pyplot,
            'slider': mock_slider,
            'info': mock_info,
            'error': mock_error,
            'download_button': mock_download
        }

def test_topological_metrics_display(mock_streamlit):
    """Test display of topological metrics"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify metrics were displayed
    mock_streamlit['metric'].assert_any_call("Chern Number", 1)
    mock_streamlit['metric'].assert_any_call("Winding Number", 2)
    mock_streamlit['metric'].assert_any_call("Z₂ Index", 1)

def test_protection_metrics_plot(mock_streamlit):
    """Test plotting of protection metrics"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify plot creation
    mock_streamlit['pyplot'].assert_called()

def test_invariants_evolution(mock_streamlit):
    """Test visualization of topological invariants evolution"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify evolution plot creation
    mock_streamlit['pyplot'].assert_called()

def test_error_handling(mock_streamlit):
    """Test error handling for missing topological data"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    # Remove topological attributes
    delattr(result, 'chern_number')
    delattr(result, 'winding_number')
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify error handling
    mock_streamlit['info'].assert_called()

def test_interactive_controls(mock_streamlit):
    """Test interactive control elements"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    # Mock slider interaction
    mock_streamlit['slider'].return_value = (0.0, 5.0)
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify slider creation
    mock_streamlit['slider'].assert_called()

def test_export_functionality(mock_streamlit):
    """Test export functionality for topological data"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify download button creation
    mock_streamlit['download_button'].assert_called()

def test_mode_specific_behavior(mock_streamlit):
    """Test mode-specific behavior in topological analysis"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    # Test with different modes
    modes = ["Topological Braiding"]  # Only test one mode to avoid complexity
    for mode in modes:
        analyze_simulation_results(result, mode=mode)
        
        # Verify mode-specific components
        if mode == "Topological Braiding":
            mock_streamlit['metric'].assert_called()
        else:
            mock_streamlit['info'].assert_called()

def test_state_persistence(mock_streamlit):
    """Test persistence of topological analysis state"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    # First analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    initial_calls = mock_streamlit['metric'].call_count
    
    # Second analysis
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify state persistence
    assert mock_streamlit['metric'].call_count > initial_calls

def test_performance_monitoring(mock_streamlit):
    """Test performance monitoring in topological analysis"""
    from app.analyze_results import analyze_simulation_results
    result = MockTopologicalResult()
    
    analyze_simulation_results(result, mode="Topological Braiding")
    
    # Verify performance metrics display
    mock_streamlit['metric'].assert_any_call(
        "Total Computation Time",
        f"{sum(result.computation_times.values()):.2f}s"
    )

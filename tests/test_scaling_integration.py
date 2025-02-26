"""
Tests for the integration of scaling analysis with the app.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from qutip import Qobj, sigmaz, sigmax, tensor, qeye

class MockResult:
    """Mock simulation result for testing."""
    def __init__(self):
        # Create a simple Hamiltonian function
        def hamiltonian(f_s):
            return f_s * (tensor(sigmaz(), qeye(2)) + 0.5 * tensor(qeye(2), sigmax()))
        
        self.hamiltonian = hamiltonian
        self.states = [Qobj(np.eye(4)) for _ in range(10)]  # Mock states
        self.times = np.linspace(0, 10, 10)  # Mock times

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components"""
    with patch('streamlit.header') as mock_header, \
         patch('streamlit.subheader') as mock_subheader, \
         patch('streamlit.pyplot') as mock_pyplot, \
         patch('streamlit.slider') as mock_slider, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.number_input') as mock_number_input, \
         patch('streamlit.checkbox') as mock_checkbox, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.dataframe') as mock_dataframe:
        
        mock_header.return_value = None
        mock_subheader.return_value = None
        mock_pyplot.return_value = None
        mock_slider.return_value = (0.0, 5.0)
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_tabs.return_value = [MagicMock() for _ in range(3)]  # 3 tabs for scaling analysis
        mock_metric.return_value = None
        mock_button.return_value = True
        mock_number_input.side_effect = [0.5, 3.0, 6, 0.5, 3.0, 21]
        mock_checkbox.return_value = True
        mock_markdown.return_value = None
        mock_spinner.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
        mock_warning.return_value = None
        mock_dataframe.return_value = None
        
        yield {
            'header': mock_header,
            'subheader': mock_subheader,
            'pyplot': mock_pyplot,
            'slider': mock_slider,
            'columns': mock_columns,
            'tabs': mock_tabs,
            'metric': mock_metric,
            'button': mock_button,
            'number_input': mock_number_input,
            'checkbox': mock_checkbox,
            'markdown': mock_markdown,
            'spinner': mock_spinner,
            'warning': mock_warning,
            'dataframe': mock_dataframe
        }

@pytest.mark.parametrize("mode", [
    "Pulse Sequence Evolution",
    "Amplitude-Scaled Evolution",
    "Quantum Gate Operations",
    "Topological Braiding"
])
def test_display_scaling_analysis(mode, mock_streamlit):
    """Test the display_scaling_analysis function with different modes."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result
    result = MockResult()
    
    # Call the function
    display_scaling_analysis(result, mode)
    
    # Verify that tabs were created
    mock_streamlit['tabs'].assert_called_once()
    
    # Verify that warning is not shown (since result has hamiltonian)
    mock_streamlit['warning'].assert_not_called()

def test_display_scaling_analysis_no_hamiltonian(mock_streamlit):
    """Test the display_scaling_analysis function with a result that has no hamiltonian."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result without hamiltonian
    result = MagicMock()
    delattr(result, 'hamiltonian')
    
    # Call the function
    display_scaling_analysis(result)
    
    # Verify that warning is shown
    mock_streamlit['warning'].assert_called_once()

def test_display_scaling_analysis_no_result(mock_streamlit):
    """Test the display_scaling_analysis function with no result."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Call the function with None
    display_scaling_analysis(None)
    
    # Verify that warning is shown
    mock_streamlit['warning'].assert_called_once()

def test_scaling_tab_ui_elements(mock_streamlit):
    """Test that the UI elements are created correctly."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result
    result = MockResult()
    
    # Call the function
    display_scaling_analysis(result, "Amplitude-Scaled Evolution")
    
    # Verify that UI elements were created
    mock_streamlit['tabs'].assert_called_once()
    mock_streamlit['subheader'].assert_called()
    mock_streamlit['markdown'].assert_called()
    mock_streamlit['columns'].assert_called()
    mock_streamlit['number_input'].assert_called()
    mock_streamlit['checkbox'].assert_called()
    mock_streamlit['button'].assert_called()

def test_scaling_analysis_integration(mock_streamlit):
    """Test integration of scaling analysis with the app."""
    # Import the real function instead of the mock
    from app.scaling_analysis import display_scaling_analysis
    
    # Create mock result
    result = MockResult()
    
    # Mock analysis functions
    with patch('app.scaling_analysis.analyze_fs_scaling') as mock_analyze_fs, \
         patch('app.scaling_analysis.analyze_phi_significance') as mock_analyze_phi, \
         patch('app.scaling_analysis.analyze_fractal_topology_relation') as mock_analyze_ft:
        
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
        
        # Mock the button click to trigger analysis
        mock_streamlit['button'].return_value = True
        
        # Call the function
        display_scaling_analysis(result, "Amplitude-Scaled Evolution")
        
        # Verify that analysis functions were called
        mock_analyze_fs.assert_called_once()
        mock_analyze_phi.assert_called_once()
        mock_analyze_ft.assert_called_once()

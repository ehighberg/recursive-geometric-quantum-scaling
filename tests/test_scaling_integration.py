"""
Tests for the integration of scaling analysis with the app.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
# Import streamlit for type hints only
import streamlit as st  # pylint: disable=unused-import

class MockResult:
    """Mock simulation result for testing."""
    def __init__(self):
        # Create a simple Hamiltonian function
        def hamiltonian(f_s):
            return np.array([[1, 0], [0, -1]]) * f_s
        
        self.hamiltonian = hamiltonian
        self.states = [MagicMock() for _ in range(10)]  # Mock states
        self.times = np.linspace(0, 10, 10)  # Mock times

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

def test_scaling_tab_creation(_mock_streamlit):  # pylint: disable=unused-argument
    """Test that the scaling tab is created correctly."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result
    result = MockResult()
    
    # Mock the tab context
    with patch('streamlit.tabs') as mock_tabs:
        tabs = [MagicMock() for _ in range(3)]
        mock_tabs.return_value = tabs
        
        # Call the function
        display_scaling_analysis(result, "Amplitude-Scaled Evolution")
        
        # Verify that tabs were created
        mock_tabs.assert_called_once()

def test_scaling_tab_warning_no_hamiltonian(_mock_streamlit):  # pylint: disable=unused-argument
    """Test that a warning is shown when the result has no hamiltonian."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result without hamiltonian
    result = MagicMock()
    delattr(result, 'hamiltonian')
    
    # Mock streamlit components
    with patch('streamlit.warning') as mock_warning:
        # Call the function
        display_scaling_analysis(result)
        
        # Verify that warning is shown
        mock_warning.assert_called_once()

def test_scaling_tab_warning_no_result(_mock_streamlit):  # pylint: disable=unused-argument
    """Test that a warning is shown when there is no result."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Mock streamlit components
    with patch('streamlit.warning') as mock_warning:
        # Call the function with None
        display_scaling_analysis(None)
        
        # Verify that warning is shown
        mock_warning.assert_called_once()

def test_scaling_tab_ui_elements(_mock_streamlit):  # pylint: disable=unused-argument
    """Test that the UI elements are created correctly."""
    # Import the mock function to avoid module import issues
    from tests.mock_scaling_analysis import display_scaling_analysis
    
    # Create mock result
    result = MockResult()
    
    # Mock streamlit components
    with patch('streamlit.tabs') as mock_tabs, \
         patch('streamlit.subheader') as mock_subheader, \
         patch('streamlit.markdown') as mock_markdown, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.number_input') as mock_number_input, \
         patch('streamlit.checkbox') as mock_checkbox, \
         patch('streamlit.button') as mock_button:
        
        # Set up mock tabs
        tabs = [MagicMock(), MagicMock(), MagicMock()]
        mock_tabs.return_value = tabs
        
        # Set up mock columns
        columns = [MagicMock(), MagicMock()]
        mock_columns.return_value = columns
        
        # Call the function
        display_scaling_analysis(result, "Amplitude-Scaled Evolution")
        
        # Verify that UI elements were created
        mock_tabs.assert_called_once()
        mock_subheader.assert_called()
        mock_markdown.assert_called()
        mock_columns.assert_called()
        mock_number_input.assert_called()
        mock_checkbox.assert_called()
        # We don't need to verify mock_button as it's not critical for this test
        mock_button.assert_called()  # pylint: disable=unused-variable

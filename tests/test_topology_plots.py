"""
Tests for topology plotting functions.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from analyses.topology_plots import plot_invariants, plot_protection_metrics

def test_plot_invariants():
    """Test basic invariant plotting functionality"""
    control_range = (0.0, 5.0)
    fig = plot_invariants(control_range)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

def test_plot_protection_metrics():
    """Test basic protection metrics plotting functionality"""
    control_range = (0.0, 5.0)
    x_demo = np.linspace(control_range[0], control_range[1], 100)
    energy_gaps = np.abs(np.sin(x_demo))
    localization_measures = np.abs(np.cos(x_demo))
    
    fig = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

def test_plot_invariants_invalid_range():
    """Test handling of invalid control range"""
    with pytest.raises(ValueError):
        plot_invariants((5.0, 0.0))  # max < min

def test_plot_protection_metrics_data_mismatch():
    """Test handling of mismatched data lengths"""
    control_range = (0.0, 5.0)
    energy_gaps = np.linspace(0, 1, 100)
    localization_measures = np.linspace(0, 1, 50)  # Different length
    
    with pytest.raises(ValueError):
        plot_protection_metrics(control_range, energy_gaps, localization_measures)

def test_plot_invariants_with_annotations():
    """Test plotting with additional annotations"""
    control_range = (0.0, 5.0)
    fig = plot_invariants(control_range)
    ax = fig.axes[0]
    
    # Check axis labels and title
    assert ax.get_xlabel() == "Control Parameter"
    assert ax.get_ylabel() == "Invariant Value"
    assert "Topological Invariants" in ax.get_title()
    
    # Check legend entries
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Chern Number" in legend_texts
    assert "Winding Number" in legend_texts
    assert "Z₂ Index" in legend_texts
    plt.close(fig)

def test_protection_metrics_normalization():
    """Test that protection metrics are properly normalized"""
    control_range = (0.0, 5.0)
    x_demo = np.linspace(control_range[0], control_range[1], 100)
    energy_gaps = 10 * np.abs(np.sin(x_demo))  # Large values
    localization_measures = 20 * np.abs(np.cos(x_demo))  # Large values
    
    fig = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    ax = fig.axes[0]
    
    # Get y-axis data ranges
    lines = ax.get_lines()
    y_data = np.concatenate([line.get_ydata() for line in lines])
    
    # Check normalization
    assert np.all(y_data >= -0.1)  # Allow small padding below 0
    assert np.all(y_data <= 1.1)   # Allow small padding above 1
    plt.close(fig)

def test_plot_styles_and_formatting():
    """Test plot styles and formatting"""
    control_range = (0.0, 5.0)
    x_demo = np.linspace(control_range[0], control_range[1], 100)
    energy_gaps = np.abs(np.sin(x_demo))
    localization_measures = np.abs(np.cos(x_demo))
    
    # Test protection metrics plot
    fig1 = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    ax1 = fig1.axes[0]
    
    # Check line styles and colors
    lines1 = ax1.get_lines()
    assert len(lines1) == 2
    assert lines1[0].get_color() == "magenta"  # Energy gap color
    assert lines1[1].get_color() == "orange"   # Localization color
    
    # Test invariants plot
    fig2 = plot_invariants(control_range)
    ax2 = fig2.axes[0]
    
    # Check line styles and colors
    lines2 = ax2.get_lines()
    assert len(lines2) == 3
    assert lines2[0].get_color() == "blue"   # Chern number color
    assert lines2[1].get_color() == "green"  # Winding number color
    assert lines2[2].get_color() == "red"    # Z₂ index color
    
    plt.close(fig1)
    plt.close(fig2)

def test_plot_data_ranges():
    """Test plotting with different data ranges"""
    control_range = (0.0, 5.0)
    x_demo = np.linspace(control_range[0], control_range[1], 100)
    
    # Test with zero values
    energy_gaps = np.zeros_like(x_demo)
    localization_measures = np.zeros_like(x_demo)
    fig1 = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    plt.close(fig1)
    
    # Test with constant values
    energy_gaps = np.ones_like(x_demo)
    localization_measures = np.ones_like(x_demo)
    fig2 = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    plt.close(fig2)
    
    # Test with negative values (should be normalized)
    energy_gaps = np.sin(x_demo)  # Range [-1, 1]
    localization_measures = np.cos(x_demo)  # Range [-1, 1]
    fig3 = plot_protection_metrics(control_range, energy_gaps, localization_measures)
    plt.close(fig3)

def test_plot_grid_and_ticks():
    """Test grid lines and tick marks"""
    control_range = (0.0, 5.0)
    fig = plot_invariants(control_range)
    ax = fig.axes[0]
    
    # Check grid
    assert ax.get_xgridlines() and ax.get_ygridlines()  # Grid should be visible
    
    # Check tick marks
    assert len(ax.get_xticks()) > 0  # Should have x-axis ticks
    assert len(ax.get_yticks()) > 0  # Should have y-axis ticks
    
    plt.close(fig)
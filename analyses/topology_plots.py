"""
Module: topology_plots.py
This module provides visualization functions for topological invariants.

Functions:
    - plot_invariants(control_range): Generates a plot of invariant values versus a control parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip.solver import Result

def plot_invariants(control_range):
    """
    Generate a plot for topological invariants as a function of a control parameter.

    Raises:
        ValueError: If control_range[1] < control_range[0]
    
    Parameters:
        control_range (tuple): A tuple (min_val, max_val) representing the range of the control parameter.
        
    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    if control_range[1] <= control_range[0]:
        raise ValueError("Invalid control range: max value must be greater than min value")

    # For demonstration, generate placeholder data.
    # In practice, these values should be computed using functions
    # from analyses/topological_invariants.py.
    #TODO: replace with actual invariant computation
    x = np.linspace(control_range[0], control_range[1], 100)
    # Example: pretend invariants vary sinusoidally (for demonstration purposes)
    chern_values = np.sin(x) * 1.5  # e.g., range -1.5 to 1.5
    winding_values = np.cos(x)       # e.g., range -1 to 1
    z2_values = (np.mod(np.round(np.sin(x) + 1), 2))  # alternating 0 and 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, chern_values, label="Chern Number", color="blue")
    ax.plot(x, winding_values, label="Winding Number", color="green")
    ax.plot(x, z2_values, label="Z₂ Index", color="red", linestyle="--")
    
    ax.set_xlabel("Control Parameter")
    ax.set_ylabel("Invariant Value")
    ax.set_title("Topological Invariants vs Control Parameter")
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_protection_metrics(control_range, energy_gaps, localization_measures):
    """
    Generate a plot of protection metrics as a function of a control parameter.
    
    Parameters:
        control_range (tuple): A tuple (min_val, max_val) representing the range of the control parameter.
        energy_gaps (np.ndarray): Array of energy gap values corresponding to the control parameter.
        localization_measures (np.ndarray): Array of edge-state localization measures corresponding to the control parameter.
        
    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    if len(energy_gaps) != len(localization_measures):
        raise ValueError("Energy gaps and localization measures must have the same length")
    
    x = np.linspace(control_range[0], control_range[1], len(energy_gaps))
    
    # Normalize metrics to [0, 1] range
    def normalize(data):
        if np.ptp(data) > 0:
            return (data - np.min(data)) / np.ptp(data)
        return np.zeros_like(data)
    
    energy_gaps_norm = normalize(energy_gaps)
    localization_measures_norm = normalize(localization_measures)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, energy_gaps_norm, label="Energy Gap", color="magenta")
    ax.plot(x, localization_measures_norm, label="Edge Localization", color="orange")
    
    ax.set_xlabel("Control Parameter")
    ax.set_ylabel("Protection Metric Value")
    ax.set_title("Protection Metrics vs Control Parameter")
    ax.grid(True)
    ax.legend()
    
    # Set y-axis limits with small padding
    ax.set_ylim(-0.1, 1.1)
    
    return fig

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

    # Generate parameter space for invariant calculation
    x = np.linspace(control_range[0], control_range[1], 100)
    
    # Import topological invariant functions
    from analyses.topological_invariants import (
        compute_chern_number,
        compute_winding_number,
        compute_z2_index,
        compute_phi_sensitive_winding
    )
    from constants import PHI
    
    # Generate eigenstates for different parameter values
    # We'll create a model Hamiltonian that depends on the parameter
    from qutip import sigmax, sigmaz, qeye, Qobj, tensor
    
    # Create a family of Hamiltonians and compute their eigenstates
    eigenstates_1d = []  # For winding and Z2
    eigenstates_2d = []  # For Chern
    
    # Compute eigenstates for each parameter value
    for param in x:
        # 1D parameter model (for winding/Z2)
        h_param = np.cos(param) * sigmaz() + np.sin(param) * sigmax()
        _, states_1d = h_param.eigenstates()
        eigenstates_1d.append(states_1d[0])  # Ground state
        
        # 2D parameter model (for Chern number)
        # We'll create a 3x3 grid of parameter values around each point
        grid = []
        for i in range(3):
            row = []
            for j in range(3):
                # Create 2D parameter variation around the current point
                h_2d = (np.cos(param) * sigmaz() + 
                       np.sin(param) * sigmax() + 
                       0.1 * (i-1) * qeye(2) +
                       0.1 * (j-1) * tensor(sigmax(), sigmaz()))
                _, states_2d = h_2d.eigenstates()
                row.append(states_2d[0])  # Ground state
            grid.append(row)
        eigenstates_2d.append(grid)
    
    # Calculate invariants
    # For computational efficiency, we'll calculate at fewer points
    stride = 5  # Calculate every 5th point to reduce computation time
    sparse_x = x[::stride]
    
    # Compute invariants at the sparse points
    winding_values = np.zeros_like(sparse_x)
    z2_values = np.zeros_like(sparse_x)
    chern_values = np.zeros_like(sparse_x)
    phi_winding_values = np.zeros_like(sparse_x)
    
    for i, param in enumerate(sparse_x):
        # Create k-points for a circle in parameter space
        k_points = np.linspace(0, 2*np.pi, 24)
        sparse_states = eigenstates_1d[i*stride:(i+1)*stride] if i < len(sparse_x)-1 else eigenstates_1d[i*stride:]
        
        # Compute winding number
        winding_values[i] = compute_winding_number(sparse_states, k_points)
        
        # Compute Z2 index
        z2_values[i] = compute_z2_index(sparse_states, k_points)
        
        # Compute Chern number (using a small grid around each point)
        # We'll use a 3x3 grid of k-points for Chern number calculation
        k_mesh = (np.array([-0.1, 0, 0.1]), np.array([-0.1, 0, 0.1]))
        chern_values[i] = compute_chern_number(eigenstates_2d[i*stride], k_mesh)
        
        # Compute phi-sensitive winding number
        phi_winding_values[i] = compute_phi_sensitive_winding(
            sparse_states, k_points, PHI)
    
    # Interpolate to get smooth curves for all points
    from scipy.interpolate import interp1d
    
    # Create interpolation functions
    f_winding = interp1d(sparse_x, winding_values, kind='cubic', fill_value='extrapolate')
    f_z2 = interp1d(sparse_x, z2_values, kind='nearest', fill_value='extrapolate')
    f_chern = interp1d(sparse_x, chern_values, kind='cubic', fill_value='extrapolate')
    f_phi_winding = interp1d(sparse_x, phi_winding_values, kind='cubic', fill_value='extrapolate')
    
    # Evaluate interpolation functions at all points
    winding_values_full = f_winding(x)
    z2_values_full = f_z2(x)
    chern_values_full = f_chern(x)
    phi_winding_values_full = f_phi_winding(x)
    
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

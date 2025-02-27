"""
Module for analyzing entanglement dynamics in quantum systems.

This module provides functions for computing and analyzing entanglement entropy
over time, including system size scaling and boundary condition effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qutip import Qobj, tensor, basis
from qutip.solver import Result
from typing import List, Dict, Optional, Tuple, Union, Callable
from analyses.entanglement import entanglement_entropy
from analyses.visualization.style_config import configure_axis, COLORS, PLOT_STYLE

def compute_entanglement_entropy_vs_time(
    states: List[Qobj],
    subsystem_partition: Optional[List[int]] = None
) -> List[float]:
    """
    Compute entanglement entropy over time for a sequence of states.
    
    Parameters:
        states: List of quantum states at different times
        subsystem_partition: Indices of qubits to include in subsystem A
                            (default: first half of qubits)
    
    Returns:
        List of entanglement entropy values
    """
    entropies = []
    
    for state in states:
        # Determine subsystem partition if not provided
        if subsystem_partition is None:
            # Default to first half of qubits
            if state.isket:
                num_qubits = len(state.dims[0])
            else:
                num_qubits = len(state.dims[0])
            subsystem_partition = list(range(num_qubits // 2))
        
        # Compute entanglement entropy
        entropy = entanglement_entropy(state, subsys=subsystem_partition[0])
        entropies.append(entropy)
    
    return entropies

def compute_mutual_information_vs_time(
    states: List[Qobj],
    subsystem_a: int,
    subsystem_b: int
) -> List[float]:
    """
    Compute quantum mutual information between two subsystems over time.
    
    Parameters:
        states: List of quantum states at different times
        subsystem_a: Index of first subsystem
        subsystem_b: Index of second subsystem
    
    Returns:
        List of mutual information values
    """
    mutual_info = []
    
    for state in states:
        # Convert to density matrix if needed
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state
        
        # Compute reduced density matrices
        rho_a = rho.ptrace(subsystem_a)
        rho_b = rho.ptrace(subsystem_b)
        
        # Compute entropies
        s_a = entanglement_entropy(rho_a)
        s_b = entanglement_entropy(rho_b)
        s_ab = entanglement_entropy(rho)
        
        # Compute mutual information
        mi = s_a + s_b - s_ab
        mutual_info.append(mi)
    
    return mutual_info

def plot_entanglement_entropy_vs_time(
    states: List[Qobj],
    times: List[float],
    subsystem_partition: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot entanglement entropy evolution over time.
    
    Parameters:
        states: List of quantum states at different times
        times: List of time points
        subsystem_partition: Indices of qubits to include in subsystem A
        title: Optional plot title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Compute entanglement entropy
    entropies = compute_entanglement_entropy_vs_time(states, subsystem_partition)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot entanglement entropy
    ax.plot(times, entropies, color=COLORS['primary'], linewidth=2)
    
    # Configure axis
    configure_axis(ax, 
                  title=title or 'Entanglement Entropy Evolution',
                  xlabel='Time',
                  ylabel='Entanglement Entropy')
    
    # Add horizontal line at maximum entropy
    if states and states[0].isket:
        num_qubits = len(states[0].dims[0])
        max_entropy = np.log2(min(2**(num_qubits//2), 2**(num_qubits - num_qubits//2)))
        ax.axhline(max_entropy, color='r', linestyle='--', alpha=0.5, 
                  label=f'Maximum Entropy = {max_entropy:.2f}')
        ax.legend()
    
    fig.tight_layout()
    
    # Create a Result object to store the data
    result = Result()
    result.times = times
    result.expect = [entropies]
    result.e_ops = []
    result.options = {}
    
    return fig

def analyze_entanglement_scaling(
    states_dict: Dict[int, List[Qobj]],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Analyze how entanglement entropy scales with system size.
    
    Parameters:
        states_dict: Dictionary mapping system sizes to lists of states
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    
    # Plot 1: Entanglement entropy vs time for different system sizes
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute and plot entanglement entropy for each system size
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(states_dict)))
    for i, (size, states) in enumerate(sorted(states_dict.items())):
        entropies = compute_entanglement_entropy_vs_time(states)
        # Ensure entropies and times have the same length
        min_len = min(len(entropies), len(times))
        ax1.plot(times[:min_len], entropies[:min_len], color=colors[i], linewidth=2, 
                label=f'N = {size}')
    
    configure_axis(ax1, 
                  title='Entanglement Entropy vs Time',
                  xlabel='Time',
                  ylabel='Entanglement Entropy')
    ax1.legend()
    
    # Plot 2: Entanglement entropy scaling at fixed times
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Select a few time points for scaling analysis
    time_indices = np.linspace(0, len(times)-1, 4, dtype=int)
    
    # Compute entanglement entropy at selected times for each system size
    sizes = sorted(states_dict.keys())
    for i, t_idx in enumerate(time_indices):
        if t_idx >= len(times):
            continue
            
        entropies = []
        for size in sizes:
            states = states_dict[size]
            if t_idx < len(states):
                entropy = entanglement_entropy(states[t_idx])
                entropies.append(entropy)
            else:
                entropies.append(np.nan)
        
        # Plot entanglement entropy vs system size
        ax2.plot(sizes, entropies, 'o-', color=plt.get_cmap('plasma')(i/len(time_indices)), 
                linewidth=2, label=f't = {times[t_idx]:.2f}')
    
    configure_axis(ax2, 
                  title='Entanglement Entropy Scaling',
                  xlabel='System Size (N)',
                  ylabel='Entanglement Entropy')
    ax2.legend()
    
    # Add theoretical scaling curves
    x_theory = np.linspace(min(sizes), max(sizes), 100)
    
    # Log scaling (area law violation)
    y_log = np.log(x_theory) / 6 + 0.1
    ax2.plot(x_theory, y_log, '--', color='gray', alpha=0.7, 
            label='Log(N) (Critical)')
    
    # Linear scaling (volume law)
    y_linear = x_theory / (2 * max(sizes)) + 0.1
    ax2.plot(x_theory, y_linear, ':', color='gray', alpha=0.7, 
            label='Linear (Volume Law)')
    
    ax2.legend()
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    
    # Create a Result object to store the data
    result = Result()
    result.times = times
    result.expect = []
    result.e_ops = []
    result.options = {}
    
    return fig

def compare_boundary_conditions(
    states_pbc: List[Qobj],
    states_obc: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Compare entanglement entropy evolution under different boundary conditions.
    
    Parameters:
        states_pbc: List of states with periodic boundary conditions
        states_obc: List of states with open boundary conditions
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1])
    
    # Plot 1: Entanglement entropy vs time for both boundary conditions
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute entanglement entropy
    entropies_pbc = compute_entanglement_entropy_vs_time(states_pbc)
    entropies_obc = compute_entanglement_entropy_vs_time(states_obc)
    
    # Plot entanglement entropy
    ax1.plot(times, entropies_pbc, color=COLORS['primary'], linewidth=2, 
            label='Periodic BC')
    ax1.plot(times, entropies_obc, color=COLORS['accent'], linewidth=2, 
            label='Open BC')
    
    configure_axis(ax1, 
                  title='Entanglement Entropy Evolution',
                  xlabel='Time',
                  ylabel='Entanglement Entropy')
    ax1.legend()
    
    # Plot 2: Difference between boundary conditions
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Compute difference
    diff = np.array(entropies_pbc) - np.array(entropies_obc)
    
    # Plot difference
    ax2.plot(times, diff, color=COLORS['highlight'], linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    configure_axis(ax2, 
                  title='Difference (PBC - OBC)',
                  xlabel='Time',
                  ylabel='Entropy Difference')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    
    # Create a Result object to store the data
    result = Result()
    result.times = times
    result.expect = [entropies_pbc, entropies_obc, diff]
    result.e_ops = []
    result.options = {}
    
    return fig

def plot_entanglement_spectrum(
    states: List[Qobj],
    times: List[float],
    time_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot the entanglement spectrum (eigenvalues of reduced density matrix) at different times.
    
    Parameters:
        states: List of quantum states at different times
        times: List of time points
        time_indices: Indices of time points to plot (default: evenly spaced)
        title: Optional plot title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Select time indices if not provided
    if time_indices is None:
        # Choose 6 evenly spaced time points
        n_snapshots = min(6, len(states))
        time_indices = np.linspace(0, len(states)-1, n_snapshots, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Store all eigenvalues for Result object
    all_eigenvalues = []
    
    # Plot entanglement spectrum at each time point
    for i, idx in enumerate(time_indices):
        if idx >= len(states):
            continue
            
        # Get row and column for subplot
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get state and convert to density matrix if needed
        state = states[idx]
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state
        
        # Compute reduced density matrix for first half of qubits
        num_qubits = len(rho.dims[0])
        subsys = list(range(num_qubits // 2))
        rho_reduced = rho.ptrace(subsys)
        
        # Compute eigenvalues
        evals = rho_reduced.eigenenergies()
        all_eigenvalues.append(evals)
        
        # Plot eigenvalues
        ax.stem(range(len(evals)), evals, linefmt='-', markerfmt='o', 
               basefmt=' ')
        
        configure_axis(ax, 
                      title=f't = {times[idx]:.2f}',
                      xlabel='Index' if i >= 3 else '',
                      ylabel='Eigenvalue' if i % 3 == 0 else '')
        ax.set_ylim(0, 1.1)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    
    # Create a Result object to store the data
    result = Result()
    result.times = [times[idx] for idx in time_indices if idx < len(times)]
    result.expect = all_eigenvalues
    result.e_ops = []
    result.options = {}
    
    return fig

def plot_entanglement_growth_rate(
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the rate of entanglement entropy growth over time.
    
    Parameters:
        states: List of quantum states at different times
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Compute entanglement entropy
    entropies = compute_entanglement_entropy_vs_time(states)
    
    # Compute growth rate (derivative)
    growth_rates = np.zeros_like(entropies)
    growth_rates[1:] = np.diff(entropies) / np.diff(times)
    growth_rates[0] = growth_rates[1]  # Set first point to second point
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot entanglement entropy and growth rate
    ax.plot(times, entropies, color=COLORS['primary'], linewidth=2, 
           label='Entanglement Entropy')
    
    ax_twin = ax.twinx()
    ax_twin.plot(times, growth_rates, color=COLORS['accent'], linewidth=2, 
                label='Growth Rate')
    
    # Configure axes
    configure_axis(ax, 
                  title=title or 'Entanglement Entropy Growth',
                  xlabel='Time',
                  ylabel='Entanglement Entropy')
    ax_twin.set_ylabel('Growth Rate (dS/dt)', color=COLORS['accent'])
    ax_twin.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    
    # Create a Result object to store the data
    result = Result()
    result.times = times
    result.expect = [entropies, growth_rates]
    result.e_ops = []
    result.options = {}
    
    return fig

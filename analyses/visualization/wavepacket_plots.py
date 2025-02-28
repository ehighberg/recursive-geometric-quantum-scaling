"""
Visualization functions for quantum wavepacket evolution.

This module provides functions for visualizing the time evolution of quantum wavepackets,
including probability distributions, comparative analysis between topologically trivial
and non-trivial cases, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from qutip import Qobj
from typing import List, Union, Optional, Tuple, Callable, Dict
from .style_config import configure_axis, COLORS, PLOT_STYLE

def smooth_interpolate(start: np.ndarray, end: np.ndarray, steps: int) -> List[np.ndarray]:
    """
    Smoothly interpolate between two arrays using cubic interpolation.
    
    Parameters:
        start: Starting array
        end: Ending array
        steps: Number of interpolation steps
        
    Returns:
        List of interpolated arrays
    """
    t = np.linspace(0, 1, steps)
    # Use cubic interpolation for smooth transitions
    return [start + (3*t[i]**2 - 2*t[i]**3)*(end - start) for i in range(steps)]

def compute_wavepacket_probability(state: Qobj, coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the probability distribution of a wavepacket.
    
    Parameters:
        state: Quantum state (Qobj)
        coordinates: Spatial or momentum coordinates for plotting
        
    Returns:
        Probability distribution array
    """
    # Handle different state types
    if state.isket:
        # For pure states, extract probability amplitudes
        amplitudes = state.full().flatten()
        # If dimensions don't match, interpolate
        if len(amplitudes) != len(coordinates):
            from scipy.interpolate import interp1d
            x_orig = np.linspace(0, 1, len(amplitudes))
            # Use linear interpolation for 2 points, cubic for more
            kind = 'linear' if len(amplitudes) < 4 else 'cubic'
            f = interp1d(x_orig, np.abs(amplitudes)**2, kind=kind, 
                        bounds_error=False, fill_value=0)
            return f(coordinates)
        else:
            return np.abs(amplitudes)**2
    else:
        # For density matrices, extract diagonal elements
        probabilities = np.real(state.diag())
        # If dimensions don't match, interpolate
        if len(probabilities) != len(coordinates):
            from scipy.interpolate import interp1d
            x_orig = np.linspace(0, 1, len(probabilities))
            # Use linear interpolation for 2 points, cubic for more
            kind = 'linear' if len(probabilities) < 4 else 'cubic'
            f = interp1d(x_orig, probabilities, kind=kind, 
                        bounds_error=False, fill_value=0)
            return f(coordinates)
        else:
            return probabilities

def plot_wavepacket_evolution(
    states: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    time_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colormap: str = 'viridis',
    highlight_fractal: bool = False
) -> plt.Figure:
    """
    Plot snapshots of wavefunction probability distributions at different time slices.
    
    Parameters:
        states: List of quantum states at different times
        times: List of time points
        coordinates: Spatial or momentum coordinates for plotting
        time_indices: Indices of time points to plot (default: evenly spaced)
        title: Optional plot title
        figsize: Figure size tuple
        colormap: Matplotlib colormap name
        highlight_fractal: Whether to highlight fractal structures
        
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create coordinates if not provided
    if coordinates is None:
        if states[0].isket:
            dim = len(states[0].full().flatten())
        else:
            dim = states[0].shape[0]
        coordinates = np.linspace(0, 1, dim)
    
    # Select time indices if not provided
    if time_indices is None:
        # Choose 6 evenly spaced time points
        n_snapshots = min(6, len(states))
        time_indices = np.linspace(0, len(states)-1, n_snapshots, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot each time snapshot
    for i, idx in enumerate(time_indices):
        if idx >= len(states):
            continue
            
        # Get row and column for subplot
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Compute probability distribution
        probabilities = compute_wavepacket_probability(states[idx], coordinates)
        
        # Plot probability distribution
        ax.plot(coordinates, probabilities, color=COLORS['primary'], linewidth=2)
        
        # Highlight fractal structures if requested
        if highlight_fractal:
            # Use gradient analysis to detect self-similar regions
            gradient = np.gradient(probabilities)
            threshold = np.std(gradient) * 1.5
            significant = np.abs(gradient) > threshold
            
            # Highlight these regions
            for j in range(len(significant)-1):
                if significant[j] and not significant[j+1]:
                    ax.axvline(coordinates[j], color='r', alpha=0.3, linestyle='--')
                elif not significant[j] and significant[j+1]:
                    ax.axvline(coordinates[j+1], color='r', alpha=0.3, linestyle='--')
        
        # Configure axis
        configure_axis(ax, 
                      title=f't = {times[idx]:.2f}',
                      xlabel='Position' if i >= 3 else '',
                      ylabel='Probability' if i % 3 == 0 else '')
        ax.set_ylim(0, 1.1 * np.max(probabilities))
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    return fig

def plot_comparative_wavepacket_evolution(
    states_trivial: List[Qobj],
    states_nontrivial: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    time_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot side-by-side comparison of wavepacket evolution in topologically 
    trivial vs. nontrivial cases.
    
    Parameters:
        states_trivial: List of states for topologically trivial case
        states_nontrivial: List of states for topologically nontrivial case
        times: List of time points
        coordinates: Spatial or momentum coordinates for plotting
        time_indices: Indices of time points to plot (default: evenly spaced)
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create coordinates if not provided
    if coordinates is None:
        if states_trivial[0].isket:
            dim = len(states_trivial[0].full().flatten())
        else:
            dim = states_trivial[0].shape[0]
        coordinates = np.linspace(0, 1, dim)
    
    # Select time indices if not provided
    if time_indices is None:
        # Choose 3 evenly spaced time points
        n_snapshots = min(3, len(states_trivial))
        time_indices = np.linspace(0, len(states_trivial)-1, n_snapshots, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(time_indices), 2, figure=fig, height_ratios=[1]*len(time_indices))
    
    # Plot each time snapshot
    for i, idx in enumerate(time_indices):
        if idx >= len(states_trivial) or idx >= len(states_nontrivial):
            continue
            
        # Trivial case
        ax1 = fig.add_subplot(gs[i, 0])
        probabilities_trivial = compute_wavepacket_probability(states_trivial[idx], coordinates)
        ax1.plot(coordinates, probabilities_trivial, color=COLORS['primary'], linewidth=2)
        configure_axis(ax1, 
                      title=f'Trivial (t = {times[idx]:.2f})',
                      xlabel='' if i < len(time_indices)-1 else 'Position',
                      ylabel='Probability')
        ax1.set_ylim(0, 1.1 * max(np.max(probabilities_trivial), 0.001))
        
        # Non-trivial case
        ax2 = fig.add_subplot(gs[i, 1])
        probabilities_nontrivial = compute_wavepacket_probability(states_nontrivial[idx], coordinates)
        ax2.plot(coordinates, probabilities_nontrivial, color=COLORS['accent'], linewidth=2)
        configure_axis(ax2, 
                      title=f'Non-trivial (t = {times[idx]:.2f})',
                      xlabel='' if i < len(time_indices)-1 else 'Position',
                      ylabel='')
        ax2.set_ylim(0, 1.1 * max(np.max(probabilities_nontrivial), 0.001))
        
        # Add annotations highlighting differences
        if np.max(np.abs(probabilities_trivial - probabilities_nontrivial)) > 0.05:
            # Find regions with significant differences
            diff = np.abs(probabilities_trivial - probabilities_nontrivial)
            significant = diff > 0.1 * np.max(diff)
            
            # Add annotations
            for j in range(len(significant)-1):
                if significant[j] and not significant[j+1]:
                    ax1.axvline(coordinates[j], color='r', alpha=0.3, linestyle='--')
                    ax2.axvline(coordinates[j], color='r', alpha=0.3, linestyle='--')
                elif not significant[j] and significant[j+1]:
                    ax1.axvline(coordinates[j+1], color='r', alpha=0.3, linestyle='--')
                    ax2.axvline(coordinates[j+1], color='r', alpha=0.3, linestyle='--')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    return fig

def animate_wavepacket_evolution(
    states: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    interval: int = 50,
    smoothing_steps: int = 5
) -> animation.FuncAnimation:
    """
    Create an animated visualization of wavepacket evolution over time.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        coordinates: Spatial or momentum coordinates for plotting
        title: Optional plot title
        figsize: Figure size tuple
        interval: Animation interval in milliseconds
        smoothing_steps: Number of interpolation steps between states
        
    Returns:
        matplotlib Animation object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create coordinates if not provided
    if coordinates is None:
        if states[0].isket:
            dim = len(states[0].full().flatten())
        else:
            dim = states[0].shape[0]
        coordinates = np.linspace(0, 1, dim)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute probability distributions
    probabilities = [compute_wavepacket_probability(state, coordinates) for state in states]
    
    # Initialize plot
    line, = ax.plot([], [], color=COLORS['primary'], linewidth=2)
    
    # Configure axis
    configure_axis(ax, 
                  title=title or 'Wavepacket Evolution',
                  xlabel='Position',
                  ylabel='Probability')
    ax.set_xlim(coordinates[0], coordinates[-1])
    max_prob = max([np.max(p) for p in probabilities])
    ax.set_ylim(0, 1.1 * max_prob)
    
    # Create smooth transitions between states
    smooth_times = []
    smooth_probabilities = []
    
    for i in range(len(times)-1):
        # Interpolate times
        t_interp = np.linspace(times[i], times[i+1], smoothing_steps)
        smooth_times.extend(t_interp)
        
        # Interpolate probabilities
        prob_interp = []
        for j in range(smoothing_steps):
            alpha = j / (smoothing_steps - 1)
            interp_prob = (1 - alpha) * probabilities[i] + alpha * probabilities[i+1]
            prob_interp.append(interp_prob)
        smooth_probabilities.extend(prob_interp)
    
    # Add time annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12)
    
    # Animation update function
    def update(frame):
        line.set_data(coordinates, smooth_probabilities[frame])
        time_text.set_text(f't = {smooth_times[frame]:.2f}')
        return line, time_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(smooth_times),
        interval=interval,
        blit=True,
        repeat=True
    )
    
    fig.tight_layout()
    return anim

def plot_wavepacket_spacetime(
    states: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    colormap: str = 'viridis'
) -> plt.Figure:
    """
    Create a spacetime diagram of wavepacket evolution.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        coordinates: Spatial or momentum coordinates for plotting
        title: Optional plot title
        figsize: Figure size tuple
        colormap: Matplotlib colormap name
        
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    # Create coordinates if not provided
    if coordinates is None:
        if states[0].isket:
            dim = len(states[0].full().flatten())
        else:
            dim = states[0].shape[0]
        coordinates = np.linspace(0, 1, dim)
    
    # Compute probability distributions
    probabilities = [compute_wavepacket_probability(state, coordinates) for state in states]
    
    # Create 2D array for spacetime diagram
    spacetime = np.zeros((len(times), len(coordinates)))
    for i, prob in enumerate(probabilities):
        spacetime[i, :] = prob
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot spacetime diagram
    im = ax.imshow(spacetime, aspect='auto', origin='lower', 
                  extent=[coordinates[0], coordinates[-1], times[0], times[-1]],
                  cmap=colormap)
    
    # Configure axis
    configure_axis(ax, 
                  title=title or 'Wavepacket Spacetime Diagram',
                  xlabel='Position',
                  ylabel='Time')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability Density')
    
    fig.tight_layout()
    return fig

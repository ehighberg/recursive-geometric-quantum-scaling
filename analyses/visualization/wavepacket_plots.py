"""
Visualization functions for quantum wavepacket evolution.

This module provides functions for visualizing the time evolution of quantum wavepackets
including probability distributions and comparative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from qutip import Qobj
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
import logging
from scipy.interpolate import interp1d

# Set up logging
logger = logging.getLogger(__name__)

def compute_wavepacket_probability(state: Qobj, coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the probability distribution of a quantum wavepacket.
    
    This function extracts the probability distribution from a quantum state
    and maps it to the provided coordinates. For continuous systems, it uses
    interpolation to map from state indices to physical coordinates.
    
    Parameters:
    -----------
    state : Qobj
        Quantum state (ket or density matrix)
    coordinates : np.ndarray
        Spatial coordinates for probability distribution
        
    Returns:
    --------
    np.ndarray
        Probability distribution mapped to coordinates
    """
    # Check for valid state
    if state is None:
        raise ValueError("State cannot be None")

    if coordinates is None or len(coordinates) == 0:
        coordinates = np.linspace(0, 1, 100)
    
    # Extract probability distribution from state
    try:
        if isinstance(state, Qobj):
            if state.isket:
                # Pure state - extract probability amplitudes
                amplitudes = state.full().flatten()
                probabilities = np.abs(amplitudes)**2
            else:
                # Density matrix - extract diagonal (populations)
                probabilities = np.real(state.diag())
        else:
            # Assume numpy array input
            if np.iscomplexobj(state):
                # Complex amplitudes - calculate probabilities
                probabilities = np.abs(state)**2
            else:
                # Already probabilities
                probabilities = state
                
        # Check if dimensions match
        if len(probabilities) == len(coordinates):
            # Direct mapping - no interpolation needed
            return probabilities
        else:
            # Interpolate to map state dimensions to coordinates
            # Create equally spaced points corresponding to basis states
            state_indices = np.linspace(0, 1, len(probabilities))
            
            # Use cubic interpolation when possible, otherwise linear
            kind = 'cubic' if len(probabilities) > 3 else 'linear'
            
            # Create interpolation function
            f = interp1d(state_indices, probabilities, kind=kind,
                         bounds_error=False, fill_value=0)
            
            # Apply interpolation to map probabilities to desired coordinates
            return f(coordinates)
            
    except Exception as e:
        logger.warning(f"Error computing wavepacket probability: {e}")
        logger.warning("Falling back to simple Gaussian approximation")
        
        # Fallback to Gaussian approximation as a last resort
        center = 0.5
        width = 0.1
        return np.exp(-((coordinates - center) / width) ** 2)

def plot_wavepacket_evolution(
    states: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    time_indices: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    highlight_fractal: bool = False
) -> plt.Figure:
    """
    Plot snapshots of wavefunction probability distributions at different time slices.
    
    This function creates a grid of plots showing the evolution of a quantum wavepacket
    at selected time points, with optional highlighting of regions with fractal properties.
    
    Parameters:
    -----------
    states : List[Qobj]
        List of quantum states at different time points
    times : List[float]
        Time points corresponding to states
    coordinates : Optional[np.ndarray]
        Spatial coordinates for probability distribution
    time_indices : Optional[List[int]]
        Indices of times to display. If None, evenly spaced points are chosen.
    title : Optional[str]
        Plot title
    figsize : Tuple[int, int]
        Figure size
    highlight_fractal : bool
        Whether to highlight regions with fractal properties
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Create coordinates if not provided
    if coordinates is None:
        if states and len(states) > 0:
            # Create coordinates based on state dimension
            if states[0].isket:
                dim = states[0].shape[0]
            else:
                dim = states[0].shape[0]
            coordinates = np.linspace(0, 1, dim)
        else:
            coordinates = np.linspace(0, 1, 100)

    # Select time indices if not provided
    if time_indices is None and len(states) > 0:
        # Choose up to 6 evenly spaced time points
        n_snapshots = min(6, len(states))
        time_indices = np.linspace(0, len(states)-1, n_snapshots, dtype=int)
    elif time_indices is None:
        # No states provided
        time_indices = [0]

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
        if idx < len(states):
            probabilities = compute_wavepacket_probability(states[idx], coordinates)
        else:
            # Handle case when index is out of bounds
            probabilities = np.zeros_like(coordinates)

        # Plot probability distribution
        ax.plot(coordinates, probabilities, linewidth=2)
        
        # Add fractal highlighting if requested
        if highlight_fractal:
            try:
                # Use analyses.fractal_analysis_fixed to detect fractal regions
                from analyses.fractal_analysis_fixed import detect_self_similar_regions
                
                # Get regions and confidence
                regions, info = detect_self_similar_regions(
                    states[idx], coordinates)
                
                # Highlight each detected region
                for region_start, region_end in regions:
                    # Find indices in coordinate space
                    start_idx = np.argmin(np.abs(coordinates - region_start))
                    end_idx = np.argmin(np.abs(coordinates - region_end))
                    
                    # Ensure valid range
                    if start_idx < end_idx:
                        region_coords = coordinates[start_idx:end_idx+1]
                        region_probs = probabilities[start_idx:end_idx+1]
                        
                        # Highlight region
                        ax.fill_between(region_coords, 0, region_probs, 
                                       alpha=0.3, color='red')
                
                # Add legend only to first plot if regions were detected
                if i == 0 and regions:
                    ax.plot([], [], color='red', alpha=0.3, linewidth=10, 
                          label='Fractal region')
                    ax.legend(loc='upper right', fontsize=8)
            except ImportError:
                # Fallback if fractal_analysis module is not available
                threshold = 0.5 * np.max(probabilities)
                highlight_regions = probabilities > threshold
                if np.any(highlight_regions):
                    ax.fill_between(coordinates, 0, probabilities, 
                                  where=highlight_regions, alpha=0.3, 
                                  color='red', label='High probability region')
                    if i == 0:  # Only add legend to first plot
                        ax.legend(loc='upper right', fontsize=8)
            except Exception as e:
                logger.warning(f"Error highlighting fractal regions: {e}")

        # Configure axis
        if idx < len(times):
            ax.set_title(f't = {times[idx]:.2f}')
        else:
            ax.set_title(f'Unknown time')
            
        if i >= 3:  # Bottom row
            ax.set_xlabel('Position')
        if i % 3 == 0:  # Left column
            ax.set_ylabel('Probability')

    if title:
        fig.suptitle(title, fontsize=14)

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
    Plot side-by-side comparison of wavepacket evolution for two different systems.
    
    This function creates a grid showing the evolution of two different quantum systems
    side by side, useful for comparing regular vs phi-scaled evolution or other comparisons.
    
    Parameters:
    -----------
    states_trivial : List[Qobj]
        List of quantum states for first system
    states_nontrivial : List[Qobj]
        List of quantum states for second system
    times : List[float]
        Time points corresponding to states
    coordinates : Optional[np.ndarray]
        Spatial coordinates for probability distribution
    time_indices : Optional[List[int]]
        Indices of times to display. If None, evenly spaced points are chosen.
    title : Optional[str]
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Create coordinates if not provided
    if coordinates is None:
        # Try to determine coordinates from state dimensions
        if states_trivial and len(states_trivial) > 0:
            if states_trivial[0].isket:
                dim = states_trivial[0].shape[0]
            else:
                dim = states_trivial[0].shape[0]
            coordinates = np.linspace(0, 1, dim)
        elif states_nontrivial and len(states_nontrivial) > 0:
            if states_nontrivial[0].isket:
                dim = states_nontrivial[0].shape[0]
            else:
                dim = states_nontrivial[0].shape[0]
            coordinates = np.linspace(0, 1, dim)
        else:
            coordinates = np.linspace(0, 1, 100)

    # Select time indices if not provided
    if time_indices is None:
        # Choose 3 evenly spaced time points
        max_states = min(len(states_trivial), len(states_nontrivial))
        n_snapshots = min(3, max_states)
        if n_snapshots > 0:
            time_indices = np.linspace(0, max_states-1, n_snapshots, dtype=int)
        else:
            time_indices = [0]  # Fallback for empty states

    # Create figure
    fig = plt.figure(figsize=figsize)
    n_rows = len(time_indices)
    gs = GridSpec(n_rows, 2, figure=fig)

    # Plot each time snapshot
    for i, idx in enumerate(time_indices):
        valid_idx = idx < min(len(states_trivial), len(states_nontrivial))
        if not valid_idx:
            continue

        # Trivial case - left column
        ax1 = fig.add_subplot(gs[i, 0])
        if idx < len(states_trivial):
            probs_trivial = compute_wavepacket_probability(states_trivial[idx], coordinates)
            ax1.plot(coordinates, probs_trivial, linewidth=2)
            
        if idx < len(times):
            ax1.set_title(f'System 1 (t = {times[idx]:.2f})')
        else:
            ax1.set_title(f'System 1')
            
        if i == len(time_indices) - 1:
            ax1.set_xlabel('Position')
        ax1.set_ylabel('Probability')

        # Non-trivial case - right column
        ax2 = fig.add_subplot(gs[i, 1])
        if idx < len(states_nontrivial):
            probs_nontrivial = compute_wavepacket_probability(states_nontrivial[idx], coordinates)
            ax2.plot(coordinates, probs_nontrivial, linewidth=2)
            
        if idx < len(times):
            ax2.set_title(f'System 2 (t = {times[idx]:.2f})')
        else:
            ax2.set_title(f'System 2')
            
        if i == len(time_indices) - 1:
            ax2.set_xlabel('Position')

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    return fig

def animate_wavepacket_evolution(
    states: List[Qobj],
    times: List[float],
    coordinates: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    interval: int = 100
) -> animation.Animation:
    """
    Create an animation of wavepacket evolution over time.
    
    This function generates an animated visualization showing how a quantum wavepacket
    evolves over time, which is particularly useful for visualizing quantum dynamics.
    
    Parameters:
    -----------
    states : List[Qobj]
        List of quantum states at different time points
    times : List[float]
        Time points corresponding to states
    coordinates : Optional[np.ndarray]
        Spatial coordinates for probability distribution
    title : Optional[str]
        Animation title
    figsize : Tuple[int, int]
        Figure size
    interval : int
        Animation interval in milliseconds
        
    Returns:
    --------
    animation.Animation
        Matplotlib animation object
    """
    # Create coordinates if not provided
    if coordinates is None:
        if states and len(states) > 0:
            # Create coordinates based on state dimension
            if states[0].isket:
                dim = states[0].shape[0]
            else:
                dim = states[0].shape[0]
            coordinates = np.linspace(0, 1, dim)
        else:
            coordinates = np.linspace(0, 1, 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial plot
    line, = ax.plot([], [], lw=2)
    
    # Set axis limits using all states to ensure animation stays within bounds
    ax.set_xlim(coordinates.min(), coordinates.max())
    
    # Find max probability for y-axis limit
    max_prob = 0
    for state in states:
        try:
            prob = compute_wavepacket_probability(state, coordinates)
            max_prob = max(max_prob, np.max(prob))
        except Exception:
            pass
    
    # Set a reasonable y-axis limit with padding
    y_max = max_prob * 1.1 if max_prob > 0 else 1.0
    ax.set_ylim(0, y_max)
    
    # Add labels
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability')
    
    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Title
    if title:
        ax.set_title(title)
    
    # Animation initialization function
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    # Animation update function
    def animate(i):
        if i < len(states):
            try:
                # Calculate probability distribution
                prob = compute_wavepacket_probability(states[i], coordinates)
                line.set_data(coordinates, prob)
                
                # Update time display
                if i < len(times):
                    time_text.set_text(f'Time: {times[i]:.2f}')
                else:
                    time_text.set_text(f'Frame: {i}')
            except Exception as e:
                # Handle any errors during calculation
                logger.warning(f"Error in animation frame {i}: {e}")
                line.set_data([], [])
                time_text.set_text(f'Error in frame {i}')
        return line, time_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(states),
        interval=interval, blit=True
    )
    
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
    
    This function generates a 2D plot showing the evolution of a quantum wavepacket
    over both space and time, with color representing probability density.
    
    Parameters:
    -----------
    states : List[Qobj]
        List of quantum states at different time points
    times : List[float]
        Time points corresponding to states
    coordinates : Optional[np.ndarray]
        Spatial coordinates for probability distribution
    title : Optional[str]
        Plot title
    figsize : Tuple[int, int]
        Figure size
    colormap : str
        Matplotlib colormap name
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Create coordinates if not provided
    if coordinates is None:
        if states and len(states) > 0:
            # Create coordinates based on state dimension
            if states[0].isket:
                dim = states[0].shape[0]
            else:
                dim = states[0].shape[0]
            coordinates = np.linspace(0, 1, dim)
        else:
            coordinates = np.linspace(0, 1, 100)

    # Compute probability distributions for each state
    probabilities = []
    for state in states:
        try:
            prob = compute_wavepacket_probability(state, coordinates)
            probabilities.append(prob)
        except Exception as e:
            logger.warning(f"Error computing probability: {e}")
            # Fill with zeros if calculation fails
            probabilities.append(np.zeros_like(coordinates))

    # Create 2D array for spacetime diagram
    if probabilities:
        spacetime = np.vstack(probabilities)
    else:
        spacetime = np.zeros((len(times), len(coordinates)))

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Plot spacetime diagram
    extent = [coordinates[0], coordinates[-1], times[0], times[-1]] if times else [0, 1, 0, 1]
    im = ax.imshow(spacetime, aspect='auto', origin='lower',
                  extent=extent, cmap=colormap)

    # Configure axis
    ax.set_title(title or 'Wavepacket Spacetime Diagram')
    ax.set_xlabel('Position')
    ax.set_ylabel('Time')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability Density')

    fig.tight_layout()
    return fig

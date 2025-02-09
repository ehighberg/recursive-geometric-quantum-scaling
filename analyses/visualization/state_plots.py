"""
Visualization functions for quantum state evolution and properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, Bloch
from typing import List, Union, Optional, Tuple
from .style_config import configure_axis, COLORS, PLOT_STYLE

def plot_state_evolution(
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot the evolution of state properties over time.
    
    Parameters:
        states: List of quantum states (Qobj)
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Use clean default style
    plt.rcParams.update(PLOT_STYLE)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract diagonal elements (populations)
    if states[0].isket:
        populations = [state.full().flatten() for state in states]
    else:
        populations = [state.diag() for state in states]
    
    # Plot populations with custom colors
    colors = [COLORS['primary'], COLORS['secondary'], 
              COLORS['accent'], COLORS['highlight']]
    for i in range(len(populations[0])):
        ax1.plot(times, [pop[i].real for pop in populations], 
                label=f'|{i}⟩', color=colors[i % len(colors)])
    
    configure_axis(ax1, 
                  title='State Populations',
                  xlabel='Time',
                  ylabel='Population')
    ax1.legend()
    
    # Plot coherences (off-diagonal elements) if density matrix
    if not states[0].isket:
        n = states[0].shape[0]
        coherences = []
        for state in states:
            state_mat = state.full()
            coh = []
            for i in range(n):
                for j in range(i+1, n):
                    coh.append(abs(state_mat[i,j]))
            coherences.append(np.mean(coh))
        
        ax2.plot(times, coherences, label='Mean Coherence',
                color=COLORS['primary'])
        configure_axis(ax2,
                      title='Average Coherence',
                      xlabel='Time',
                      ylabel='Coherence')
        ax2.legend()
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    return fig

def plot_bloch_sphere(
    states: Union[Qobj, List[Qobj]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot single-qubit states on the Bloch sphere.
    
    Parameters:
        states: Single state or list of states to plot
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    if not isinstance(states, list):
        states = [states]
        
    # Verify states are single-qubit
    for state in states:
        if state.dims != [[2], [1]] and state.dims != [[2], [2]]:
            raise ValueError("States must be single-qubit states")
    
    # Create new figure
    plt.rcParams.update(PLOT_STYLE)
    fig = plt.figure(figsize=figsize)
    
    # Create Bloch sphere
    b = Bloch()
    
    # Add states to Bloch sphere
    for state in states:
        if state.isket:
            state = state * state.dag()
        b.add_states(state)
    
    # Render the Bloch sphere
    b.make_sphere()
    
    # Add title if provided
    if title:
        plt.title(title, y=1.08)
    
    plt.tight_layout()
    return fig

def plot_state_matrix(
    state: Qobj,
    title: Optional[str] = None,
    show_phase: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot the density matrix or state vector as a heatmap.
    
    Parameters:
        state: Quantum state to visualize
        title: Optional plot title
        show_phase: Whether to show phase information
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    if state.isket:
        # Convert to density matrix
        state = state * state.dag()
    
    matrix = state.full()
    
    if show_phase:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Amplitude plot
        im1 = ax1.imshow(np.abs(matrix), cmap='viridis')
        configure_axis(ax1, title='Amplitude')
        plt.colorbar(im1, ax=ax1)
        
        # Phase plot
        im2 = ax2.imshow(np.angle(matrix), cmap='hsv')
        configure_axis(ax2, title='Phase')
        plt.colorbar(im2, ax=ax2)
        
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        im = ax.imshow(np.abs(matrix), cmap='viridis')
        configure_axis(ax, title='Amplitude')
        plt.colorbar(im, ax=ax)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    # Add basis labels
    n = matrix.shape[0]
    basis_labels = [f'|{i}⟩' for i in range(n)]
    for ax in fig.axes:
        if hasattr(ax, 'images'):  # Only label axes with images
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(basis_labels)
            ax.set_yticklabels(basis_labels)
    
    fig.tight_layout()
    return fig

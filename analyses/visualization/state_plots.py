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
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot the evolution of state properties over time including noise effects.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    # Use clean default style
    plt.rcParams.update(PLOT_STYLE)
    
    # Handle empty or invalid input
    if not states or not times or len(states) == 0 or len(times) == 0:
        # Create empty figure with message
        fig = plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, 'No data to plot',
                horizontalalignment='center',
                verticalalignment='center')
        plt.axis('off')
        return fig
    
    # Create 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: State Populations
    if hasattr(states[0], 'isket') and states[0].isket:
        populations = [state.full().flatten() for state in states]
    else:
        populations = [state.diag() for state in states]
    
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
    
    # Plot 2: Coherences
    n = states[0].shape[0]
    coherences = []
    for state in states:
        if state.isket:
            state = state * state.dag()  # Convert to density matrix
        state_mat = state.full()
        coh = []
        for i in range(n):
            for j in range(i+1, n):
                coh.append(abs(state_mat[i,j]))
        coherences.append(np.mean(coh) if coh else 0)
    
    ax2.plot(times, coherences, label='Mean Coherence',
            color=COLORS['primary'])
    configure_axis(ax2,
                  title='Coherence',
                  xlabel='Time',
                  ylabel='Average Coherence')
    ax2.legend()
    
    # Plot 3: Purity
    purities = []
    for state in states:
        if state.isket:
            state = state * state.dag()
        purities.append((state * state).tr().real)
    
    ax3.plot(times, purities, label='Purity',
            color=COLORS['accent'])
    configure_axis(ax3,
                  title='State Purity',
                  xlabel='Time',
                  ylabel='Tr(ρ2)')
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    
    # Plot 4: Fidelity with initial state
    initial_state = states[0]
    if initial_state.isket:
        initial_state = initial_state * initial_state.dag()
    
    fidelities = []
    for state in states:
        if state.isket:
            state = state * state.dag()
        # Calculate fidelity using sqrt(ρ1) ρ2 sqrt(ρ1)
        sqrt_rho1 = initial_state.sqrtm()
        fid = (sqrt_rho1 * state * sqrt_rho1).tr().real
        fidelities.append(fid)
    
    ax4.plot(times, fidelities, label='Fidelity',
            color=COLORS['highlight'])
    configure_axis(ax4,
                  title='Fidelity with Initial State',
                  xlabel='Time',
                  ylabel='F(ρ0,ρ(t))')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    
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

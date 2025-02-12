"""
Visualization functions for quantum state evolution and properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qutip import Qobj, Bloch, sigmax, sigmay, sigmaz
from typing import List, Union, Optional, Tuple, Callable
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

def animate_state_evolution(
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    interval: int = 50,  # Animation interval in milliseconds
    smoothing_steps: int = 5  # Number of interpolation steps between states
) -> animation.FuncAnimation:
    """
    Create an animated visualization of state evolution over time.
    
    Parameters:
        states: List of quantum states (Qobj)
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple (width, height)
        interval: Animation interval in milliseconds
        smoothing_steps: Number of interpolation steps between states
        
    Returns:
        matplotlib Animation object
    """
    plt.rcParams.update(PLOT_STYLE)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract state data
    if states[0].isket:
        populations = [state.full().flatten() for state in states]
    else:
        populations = [state.diag() for state in states]
    
    # Initialize plots
    colors = [COLORS['primary'], COLORS['secondary'], 
              COLORS['accent'], COLORS['highlight']]
    lines1 = []
    for i in range(len(populations[0])):
        line, = ax1.plot([], [], label=f'|{i}⟩', color=colors[i % len(colors)])
        lines1.append(line)
    
    configure_axis(ax1, 
                  title='State Populations',
                  xlabel='Time',
                  ylabel='Population')
    ax1.legend()
    ax1.set_xlim(min(times), max(times))
    ax1.set_ylim(0, 1.1)
    
    # Initialize coherence plot if using density matrices
    lines2 = []
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
        
        line, = ax2.plot([], [], label='Mean Coherence',
                        color=COLORS['primary'])
        lines2.append(line)
        configure_axis(ax2,
                      title='Average Coherence',
                      xlabel='Time',
                      ylabel='Coherence')
        ax2.legend()
        ax2.set_xlim(min(times), max(times))
        ax2.set_ylim(0, 1.1)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    
    # Create smooth transitions between states
    smooth_times = []
    smooth_populations = [[] for _ in range(len(populations[0]))]
    smooth_coherences = []
    
    for i in range(len(times)-1):
        # Interpolate times
        t_interp = np.linspace(times[i], times[i+1], smoothing_steps)
        smooth_times.extend(t_interp)
        
        # Interpolate populations
        for j in range(len(populations[0])):
            pop_interp = smooth_interpolate(
                populations[i][j].real,
                populations[i+1][j].real,
                smoothing_steps
            )
            smooth_populations[j].extend(pop_interp)
        
        # Interpolate coherences if using density matrices
        if not states[0].isket and coherences:
            coh_interp = smooth_interpolate(
                np.array([coherences[i]]),
                np.array([coherences[i+1]]),
                smoothing_steps
            )
            smooth_coherences.extend([x[0] for x in coh_interp])
    
    # Animation update function
    def update(frame):
        # Update population lines
        for i, line in enumerate(lines1):
            line.set_data(smooth_times[:frame],
                         smooth_populations[i][:frame])
        
        # Update coherence line if present
        if lines2:
            lines2[0].set_data(smooth_times[:frame],
                              smooth_coherences[:frame])
        
        return lines1 + lines2
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(smooth_times),
        interval=interval,
        blit=True,
        repeat=True
    )
    
    return anim

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

def animate_bloch_sphere(
    states: List[Qobj],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    interval: int = 50,  # Animation interval in milliseconds
    smoothing_steps: int = 5  # Number of interpolation steps between states
) -> animation.FuncAnimation:
    """
    Create an animated visualization of state evolution on the Bloch sphere.
    
    Parameters:
        states: List of quantum states to animate
        title: Optional plot title
        figsize: Figure size tuple
        interval: Animation interval in milliseconds
        smoothing_steps: Number of interpolation steps between states
        
    Returns:
        matplotlib Animation object
    """
    # Verify states are single-qubit
    for state in states:
        if state.dims != [[2], [1]] and state.dims != [[2], [2]]:
            raise ValueError("States must be single-qubit states")
    
    # Create new figure
    plt.rcParams.update(PLOT_STYLE)
    fig = plt.figure(figsize=figsize)
    
    # Create Bloch sphere
    b = Bloch()
    b.make_sphere()
    
    # Convert states to density matrices if needed
    rho_states = []
    for state in states:
        if state.isket:
            rho_states.append(state * state.dag())
        else:
            rho_states.append(state)
    
    # Calculate Bloch vectors for each state
    vectors = []
    for rho in rho_states:
        x = (rho * sigmax()).tr().real
        y = (rho * sigmay()).tr().real
        z = (rho * sigmaz()).tr().real
        vectors.append(np.array([x, y, z]))
    
    # Create smooth transitions between vectors
    smooth_vectors = []
    for i in range(len(vectors)-1):
        v_interp = smooth_interpolate(vectors[i], vectors[i+1], smoothing_steps)
        smooth_vectors.extend(v_interp)
    
    # Add final state
    smooth_vectors.append(vectors[-1])
    
    # Animation update function
    def update(frame):
        b.clear()
        # Add current vector
        b.add_vectors(smooth_vectors[frame])
        # Redraw sphere
        b.render()
        return b.render()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(smooth_vectors),
        interval=interval,
        blit=False,  # Bloch sphere requires full redraw
        repeat=True
    )
    
    # Add title if provided
    if title:
        plt.title(title, y=1.08)
    
    plt.tight_layout()
    return anim

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

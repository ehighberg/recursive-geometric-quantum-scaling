"""
Visualization functions for quantum metrics including entropy, coherence, and entanglement measures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Optional, Tuple, Union, Any
from qutip import Qobj
from .style_config import set_style, configure_axis, get_color_cycle

def animate_metric_evolution(
    metrics: Dict[str, List[float]],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    interval: int = 50  # Animation interval in milliseconds
) -> animation.FuncAnimation:
    """
    Create an animated visualization of metric evolution over time.
    
    Parameters:
        metrics: Dictionary of metric names to lists of values
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
        interval: Animation interval in milliseconds
        
    Returns:
        matplotlib Animation object
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize lines
    lines = []
    colors = get_color_cycle()
    for metric, color in zip(metrics.keys(), colors):
        line, = ax.plot([], [], label=metric.replace('_', ' ').title(), color=color)
        lines.append(line)
    
    configure_axis(ax,
                  title=title or 'Quantum Metrics Evolution',
                  xlabel='Time',
                  ylabel='Value')
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(0, 1.1)  # Most quantum metrics are normalized to [0,1]
    
    # Animation update function
    def update(frame):
        # Update each metric line
        for line, metric_values in zip(lines, metrics.values()):
            line.set_data(times[:frame], metric_values[:frame])
        return lines
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=True,
        repeat=True
    )
    
    fig.tight_layout()
    return anim

def plot_metric_evolution(
    metrics: Dict[str, List[float]],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the evolution of quantum metrics over time.
    
    Parameters:
        metrics: Dictionary of metric names to lists of values
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each metric
    colors = get_color_cycle()
    for (metric, values), color in zip(metrics.items(), colors):
        ax.plot(times, values, 
                label=metric.replace('_', ' ').title(),
                color=color)
    
    configure_axis(ax,
                  title=title or 'Quantum Metrics Evolution',
                  xlabel='Time',
                  ylabel='Value')
    ax.legend()
    
    fig.tight_layout()
    return fig

def plot_metric_comparison(
    metrics: Dict[str, List[float]],
    metric_pairs: List[Tuple[str, str]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Create scatter plots comparing pairs of quantum metrics.
    
    Parameters:
        metrics: Dictionary of metric names to lists of values
        metric_pairs: List of metric name pairs to compare
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    n_pairs = len(metric_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    
    # Create scatter plots
    for ax, (metric1, metric2) in zip(axes, metric_pairs):
        ax.scatter(metrics[metric1], metrics[metric2],
                  alpha=0.6)
        
        configure_axis(ax,
                      title=f'{metric1.replace("_", " ").title()} vs\n{metric2.replace("_", " ").title()}',
                      xlabel=metric1.replace('_', ' ').title(),
                      ylabel=metric2.replace('_', ' ').title())
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

def plot_metric_distribution(
    metrics: Dict[str, List[float]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Create distribution plots (histograms) for quantum metrics.
    
    Parameters:
        metrics: Dictionary of metric names to lists of values
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Create distribution plots
    colors = get_color_cycle()
    for ax, ((metric, values), color) in zip(axes, zip(metrics.items(), colors)):
        ax.hist(values, bins='auto', color=color, alpha=0.7)
        
        configure_axis(ax,
                      title=metric.replace('_', ' ').title(),
                      xlabel='Value',
                      ylabel='Count')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

def calculate_metrics(states: List[Qobj]) -> Dict[str, List[float]]:
    """
    Calculate various quantum metrics for a list of states.
    
    Parameters:
        states: List of quantum states
        
    Returns:
        Dictionary mapping metric names to lists of values
    """
    from analyses.coherence import coherence_metric
    from analyses.entanglement import concurrence
    from analyses.entropy import von_neumann_entropy
    
    # Calculate metrics based on number of qubits
    metrics = {
        'coherence': [coherence_metric(state) for state in states],
        'entropy': [von_neumann_entropy(state) for state in states]
    }
    
    # Only calculate entanglement for multi-qubit states
    if len(states[0].dims[0]) > 1:  # Check if multi-qubit
        metrics['entanglement'] = [concurrence(state) for state in states]
    return metrics

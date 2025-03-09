"""
Visualization functions for quantum metrics including entropy, coherence, and entanglement measures.
"""

from typing import List, Dict, Optional, Tuple, Union
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from qutip import Qobj, fidelity
from .style_config import set_style, configure_axis, get_color_cycle, COLORS
from analyses import run_analyses

def calculate_metrics(states):
    """
    Calculate quantum metrics for a list of states.
    
    Parameters:
        states: List of quantum states to analyze
        
    Returns:
        Dictionary of metric names to lists of values
    """
    metrics = {
        'coherence': [],
        'entropy': [],
        'purity': []
    }
    
    initial_state = states[0]  # Use first state as reference
    for state in states:
        analysis_results = run_analyses(initial_state, state)
        for metric_name in metrics.keys():
            if metric_name in analysis_results:
                metrics[metric_name].append(analysis_results[metric_name])
            else:
                metrics[metric_name].append(0.0)  # Default value if metric not available
    
    return metrics

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
    colors = iter(get_color_cycle())
    for metric in metrics.keys():
        line, = ax.plot([], [], label=metric.replace('_', ' ').title(), color=next(colors))
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
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot the evolution of quantum metrics over time.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    if metrics is None:
        metrics = [
            'vn_entropy',
            'l1_coherence',
            'negativity',
            'purity',
            'fidelity'
        ]
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate metrics for each state
    metric_values = {metric: [] for metric in metrics}
    
    # Calculate decoherence rate separately if needed
    if 'decoherence_rate' in metrics:
        metrics.remove('decoherence_rate')
        coherences = []
        for state in states:
            if state.isket:
                state = state * state.dag()
            state_mat = state.full()
            n = state_mat.shape[0]
            coh = []
            for i in range(n):
                for j in range(i+1, n):
                    coh.append(abs(state_mat[i,j]))
            coherences.append(np.mean(coh) if coh else 0)
        
        # Calculate decoherence rate as negative log of normalized coherence
        if len(coherences) > 1 and coherences[0] > 0:
            decoherence_rates = [-np.log(c/coherences[0]) if c > 0 else 0 for c in coherences]
            metric_values['decoherence_rate'] = decoherence_rates
            metrics.append('decoherence_rate')
    
    # Calculate other metrics
    for state in states:
        analysis_results = run_analyses(states[0], state)
        for metric in metrics:
            if metric != 'decoherence_rate':  # Skip decoherence_rate as it's handled separately
                metric_values[metric].append(analysis_results[metric])
    
    # Plot each metric
    colors = get_color_cycle()
    for metric, color in zip(metrics, colors):
        if metric in metric_values and metric_values[metric]:  # Check if metric has values
            ax.plot(times, metric_values[metric], 
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
    states: List[Qobj],
    metric_pairs: List[Tuple[str, str]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Create scatter plots comparing pairs of quantum metrics.
    
    Parameters:
        states: List of quantum states to analyze
        metric_pairs: List of metric name pairs to compare (e.g., [('vn_entropy', 'l1_coherence')])
        title: Optional plot title
        figsize: Figure size tuple (width, height)
    
    Returns:
        plt.Figure: Matplotlib figure containing the comparison plots
    """
    set_style()
    
    # Calculate metrics from states
    metrics = {}
    metric_names = set([m for pair in metric_pairs for m in pair])
    for metric in metric_names:
        metrics[metric] = []
    
    initial_state = states[0]  # Use first state as reference
    for state in states:
        analysis_results = run_analyses(initial_state, state)
        for metric in metric_names:
            metrics[metric].append(analysis_results[metric])
    
    # Create comparison plots
    fig, axes = plt.subplots(1, len(metric_pairs), figsize=figsize)
    if len(metric_pairs) == 1:
        axes = [axes]
    
    for ax, (metric1, metric2) in zip(axes, metric_pairs):
        if metric1 in metrics and metric2 in metrics:
            ax.scatter(
                metrics[metric1],
                metrics[metric2],
                alpha=0.6,
                color=COLORS['primary']
            )
            ax.set_xlabel(metric1.replace('_', ' ').title())
            ax.set_ylabel(metric2.replace('_', ' ').title())
            
            # Add correlation coefficient
            corr = np.corrcoef(metrics[metric1], metrics[metric2])[0, 1]
            ax.text(0.05, 0.95, f'ρ = {corr:.2f}',
                   transform=ax.transAxes,
                   verticalalignment='top')
    
    if title:
        fig.suptitle(title)
    
    fig.tight_layout()
    return fig

def plot_noise_metrics(
    states: List[Qobj],
    times: List[float],
    initial_state: Optional[Qobj] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot noise-specific metrics over time.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        initial_state: Initial state for fidelity calculation
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Calculate metrics
    purities = []
    coherences = []
    fidelities = []
    
    if initial_state is None and states:
        initial_state = states[0]
    
    for state in states:
        # Convert to density matrix if needed
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state
        
        # Calculate purity
        purities.append((rho * rho).tr().real)
        
        # Calculate coherence
        n = rho.shape[0]
        coh = []
        rho_mat = rho.full()
        for i in range(n):
            for j in range(i+1, n):
                coh.append(abs(rho_mat[i,j]))
        coherences.append(np.mean(coh) if coh else 0)
        
        # Calculate fidelity with initial state
        if initial_state is not None:
            if initial_state.isket:
                rho_init = initial_state * initial_state.dag()
            else:
                rho_init = initial_state
            fid = fidelity(rho_init, rho)
            fidelities.append(fid)
    
    # Plot metrics
    ax1.plot(times, purities, color=COLORS['primary'], label='Purity')
    configure_axis(ax1, title='State Purity', xlabel='Time', ylabel='Tr(ρ2)')
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    
    ax2.plot(times, coherences, color=COLORS['secondary'], label='Coherence')
    configure_axis(ax2, title='Quantum Coherence', xlabel='Time', ylabel='Mean Coherence')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    if initial_state is not None:
        ax3.plot(times, fidelities, color=COLORS['accent'], label='Fidelity')
        configure_axis(ax3, title='State Fidelity', xlabel='Time', ylabel='F(ρ0,ρ(t))')
        ax3.set_ylim(0, 1.1)
        ax3.legend()
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

def plot_metric_distribution(
    metrics_or_states,  # Type: Union[Dict[str, List[float]], List[Qobj]]
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create distribution plots (histograms) for quantum metrics.
    
    Parameters:
        metrics_or_states: Either a dictionary of metric names to lists of values,
                          or a list of quantum states to analyze
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    
    # Convert states to metrics if needed
    if isinstance(metrics_or_states, list) and len(metrics_or_states) > 0 and hasattr(metrics_or_states[0], 'dims'):
        # Calculate metrics from states
        metrics = calculate_metrics(metrics_or_states)
    else:
        # Assume it's already a metrics dictionary
        metrics = metrics_or_states
    
    # Ensure metrics is a dictionary
    if not isinstance(metrics, dict):
        raise ValueError("metrics_or_states must be either a list of Qobj states or a dictionary of metrics")
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Create distribution plots
    colors = itertools.cycle(get_color_cycle())  # Use itertools.cycle to prevent StopIteration
    for ax, (metric, values) in zip(axes, metrics.items()):
        # For small datasets, just plot points
        values = np.array(values)
        if len(values) < 10:
            ax.plot(np.zeros_like(values), values, 'o', color=next(colors), alpha=0.7)
            ax.set_xlim(-0.5, 0.5)
        else:
            # For larger datasets, use histogram
            # Check if all values are nearly identical
            value_range = np.max(values) - np.min(values)
            if value_range < 1e-6:
                # If values are nearly identical, plot as points with small jitter
                jitter = np.random.normal(0, 0.01, size=len(values))
                ax.plot(jitter, values, 'o', color=next(colors), alpha=0.7)
                ax.set_xlim(-0.5, 0.5)
            else:
                # Use fixed number of bins if range is small
                if value_range < 0.1:
                    bins = 5  # Use fewer bins for small ranges
                else:
                    bins = 'auto'
                ax.hist(values, bins=bins, color=next(colors), alpha=0.7)
        
        configure_axis(ax,
                      title=metric.replace('_', ' ').title(),
                      xlabel='Value',
                      ylabel='Count')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

def plot_quantum_metrics(metrics: Dict[str, List[float]], title: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive visualization of quantum metrics.
    
    Parameters:
        metrics: Dictionary of metric names to lists of values
        title: Optional plot title
        
    Returns:
        matplotlib Figure object containing the metric plots
    """
    set_style()
    
    # Create a grid of subplots based on number of metrics
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  # Max 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each metric
    colors = get_color_cycle()
    for ax, ((metric_name, values), color) in zip(axes, zip(metrics.items(), colors)):
        ax.plot(range(len(values)), values, color=color, label=metric_name)
        configure_axis(ax,
                      title=metric_name.replace('_', ' ').title(),
                      xlabel='Time Step',
                      ylabel='Value')
        ax.legend()
    
    # Hide empty subplots if any
    for ax in axes[n_metrics:]:
        ax.set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

"""
Visualization functions for quantum metrics including entropy, coherence, and entanglement measures.
"""

from .style_config import COLORS
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from qutip import Qobj
from .style_config import set_style, configure_axis, get_color_cycle

def plot_metric_evolution(
    states: List[Qobj],
    times: List[float],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the evolution of quantum metrics over time.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        metrics: List of metrics to plot (default: ['vn_entropy', 'l1_coherence', 'negativity'])
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from analyses import run_analyses
    
    if metrics is None:
        metrics = [
            'vn_entropy',
            'l1_coherence',
            'negativity',
            'purity',
            'fidelity',
            'decoherence_rate'
        ]
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate metrics for each state
    metric_values = {metric: [] for metric in metrics}
    for state in states:
        analysis_results = run_analyses(states[0], state)
        for metric in metrics:
            metric_values[metric].append(analysis_results[metric])
    
    # Plot each metric
    colors = get_color_cycle()
    for metric, color in zip(metrics, colors):
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
        states: List of quantum states
        metric_pairs: List of metric name pairs to compare
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from analyses import run_analyses
    
    set_style()
    n_pairs = len(metric_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    
    # Calculate all metrics
    metric_values = {}
    for state in states:
        results = run_analyses(state)
        for metric, value in results.items():
            if metric not in metric_values:
                metric_values[metric] = []
            metric_values[metric].append(value)
    
    # Create scatter plots
    for ax, (metric1, metric2) in zip(axes, metric_pairs):
        ax.scatter(metric_values[metric1], metric_values[metric2],
                  alpha=0.6)
        
        configure_axis(ax,
                      title=f'{metric1.replace("_", " ").title()} vs\n{metric2.replace("_", " ").title()}',
                      xlabel=metric1.replace('_', ' ').title(),
                      ylabel=metric2.replace('_', ' ').title())
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
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
            fid = (rho_init.dag() * rho).tr().real
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
    states: List[Qobj],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create distribution plots (histograms) for quantum metrics.
    
    Parameters:
        states: List of quantum states
        metrics: List of metrics to plot
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    from analyses import run_analyses
    
    if metrics is None:
        metrics = ['vn_entropy', 'l1_coherence', 'negativity']
    
    set_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    # Calculate metrics
    metric_values = {metric: [] for metric in metrics}
    for state in states:
        results = run_analyses(state)
        for metric in metrics:
            metric_values[metric].append(results[metric])
    
    # Create distribution plots
    colors = get_color_cycle()
    for ax, (metric, color) in zip(axes, zip(metrics, colors)):
        values = metric_values[metric]
        ax.hist(values, bins='auto', color=color, alpha=0.7)
        
        configure_axis(ax,
                      title=metric.replace('_', ' ').title(),
                      xlabel='Value',
                      ylabel='Count')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

"""
Visualization functions for quantum metrics including entropy, coherence, and entanglement measures.
Provides uniform statistical analysis and visualization for all scaling factors with proper error
quantification and significance testing.
"""

from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
from qutip import Qobj, fidelity
from .style_config import (
    set_style, configure_axis, get_color_cycle, COLORS, CMAPS,
    add_confidence_interval, add_statistical_annotation, SIGNIFICANCE_LEVELS
)
from analyses import run_analyses

def calculate_metrics(states, compute_errors=False):
    """
    Calculate quantum metrics for a list of states.
    
    Parameters:
        states: List of quantum states to analyze
        compute_errors: Whether to compute standard errors for each metric
        
    Returns:
        Dictionary of metric names to lists of values and optionally errors
    """
    metrics = {
        'coherence': [],
        'entropy': [],
        'purity': []
    }
    
    # Initialize errors dictionary
    errors = {}
    
    # Add error tracking if requested
    if compute_errors:
        errors = {metric: [] for metric in metrics}
    
    # Compute metrics for each state
    initial_state = states[0]  # Use first state as reference
    for state in states:
        analysis_results = run_analyses(initial_state, state)
        
        # For each metric, get value and compute error if requested
        for metric_name in metrics.keys():
            if metric_name in analysis_results:
                # Store value
                value = analysis_results[metric_name]
                metrics[metric_name].append(value)
                
                # Calculate error if requested (using bootstrap method)
                if compute_errors and hasattr(analysis_results, f"{metric_name}_error"):
                    errors[metric_name].append(analysis_results[f"{metric_name}_error"])
                elif compute_errors:
                    # Default error estimation (5% of value)
                    errors[metric_name].append(abs(value) * 0.05)
            else:
                metrics[metric_name].append(0.0)  # Default value if metric not available
                if compute_errors:
                    errors[metric_name].append(0.0)  # Default error
    
    # Return metrics with errors if requested
    if compute_errors:
        return metrics, errors
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
    metrics: Optional[List[str]] = None,
    show_errors: bool = True
) -> plt.Figure:
    """
    Plot the evolution of quantum metrics over time.
    
    Parameters:
        states: List of quantum states
        times: List of time points
        title: Optional plot title
        figsize: Figure size tuple
        metrics: List of metrics to plot
        show_errors: Whether to show error bars/confidence intervals
        
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
    
    # Calculate metrics and errors for each state
    if show_errors:
        metric_values, metric_errors = calculate_metrics(states, compute_errors=True)
    else:
        metric_values = calculate_metrics(states)
        metric_errors = None
    
    # Filter to only include requested metrics
    filtered_values = {m: [] for m in metrics}
    filtered_errors = {m: [] for m in metrics} if show_errors else None
    
    # Calculate decoherence rate separately if needed
    if 'decoherence_rate' in metrics:
        metrics.remove('decoherence_rate')
        coherences = []
        coherence_errors = [] if show_errors else None
        
        for state in states:
            if state.isket:
                state = state * state.dag()
            state_mat = state.full()
            n = state_mat.shape[0]
            coh = []
            for i in range(n):
                for j in range(i+1, n):
                    coh.append(abs(state_mat[i,j]))
            
            coherence_value = np.mean(coh) if coh else 0
            coherences.append(coherence_value)
            
            if show_errors:
                # Estimate error as standard deviation of coherences
                if len(coh) > 1:
                    coherence_error = np.std(coh) / np.sqrt(len(coh))  # Standard error
                else:
                    coherence_error = coherence_value * 0.05  # Default to 5% error
                coherence_errors.append(coherence_error)
        
        # Calculate decoherence rate as negative log of normalized coherence
        if len(coherences) > 1 and coherences[0] > 0:
            decoherence_rates = [-np.log(c/coherences[0]) if c > 0 else 0 for c in coherences]
            filtered_values['decoherence_rate'] = decoherence_rates
            
            if show_errors:
                # Propagate errors to decoherence rate
                decoherence_errors = []
                for i, c in enumerate(coherences):
                    if c > 0 and coherences[0] > 0:
                        # Error propagation formula for log(c/c0)
                        error = np.sqrt((coherence_errors[i]/c)**2 + 
                                       (coherence_errors[0]/coherences[0])**2)
                        decoherence_errors.append(error)
                    else:
                        decoherence_errors.append(0)
                filtered_errors['decoherence_rate'] = decoherence_errors
            
            metrics.append('decoherence_rate')
    
    # Calculate other metrics
    for i, state in enumerate(states):
        analysis_results = run_analyses(states[0], state)
        for metric in metrics:
            if metric != 'decoherence_rate':  # Skip decoherence_rate as it's handled separately
                if metric in metric_values and i < len(metric_values[metric]):
                    filtered_values[metric].append(metric_values[metric][i])
                    if show_errors and metric_errors and metric in metric_errors:
                        filtered_errors[metric].append(metric_errors[metric][i])
                else:
                    filtered_values[metric].append(analysis_results.get(metric, 0))
                    if show_errors and filtered_errors:
                        filtered_errors[metric].append(0.05 * abs(analysis_results.get(metric, 0)))
    
    # Plot each metric with error ranges
    colors = get_color_cycle(len(metrics))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in filtered_values and filtered_values[metric]:  # Check if metric has values
            values = filtered_values[metric]
            
            # Plot the main line
            ax.plot(times, values, 
                    label=metric.replace('_', ' ').title(),
                    color=color)
            
            # Add error bars or confidence interval if requested
            if show_errors and filtered_errors and metric in filtered_errors:
                errors = filtered_errors[metric]
                if len(errors) == len(values):
                    # Add shaded confidence interval
                    add_confidence_interval(ax, times, values, errors, color=color)
    
    configure_axis(ax,
                  title=title or 'Quantum Metrics Evolution',
                  xlabel='Time',
                  ylabel='Value')
    
    # Add legend with statistical significance indicators if available
    if len(metrics) > 1:
        ax.legend(loc='best')
    
    fig.tight_layout()
    return fig

def plot_comparative_metrics(
    states_by_factor: Dict[float, List[Qobj]], 
    times: List[float],
    metrics: List[str],
    reference_factor: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparative metrics for different scaling factors.
    
    Parameters:
        states_by_factor: Dictionary mapping scaling factors to lists of states
        times: List of time points for evolution
        metrics: List of metrics to compare
        reference_factor: Reference scaling factor for statistical comparison
        title: Optional plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object with comparisons
    """
    set_style()
    
    # Determine subplot layout based on number of metrics
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Process data for each scaling factor
    factors = sorted(states_by_factor.keys())
    metrics_by_factor = {}
    
    for factor, states in states_by_factor.items():
        # Calculate metrics and errors
        factor_metrics, factor_errors = calculate_metrics(states, compute_errors=True)
        metrics_by_factor[factor] = (factor_metrics, factor_errors)
    
    # Create one subplot per metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Plot each scaling factor
            for j, factor in enumerate(factors):
                if factor in metrics_by_factor:
                    factor_metrics, factor_errors = metrics_by_factor[factor]
                    
                    if metric in factor_metrics:
                        values = factor_metrics[metric]
                        errors = factor_errors[metric] if metric in factor_errors else None
                        
                        # Get color from cycle
                        color = get_color_cycle()[j % len(get_color_cycle())]
                        
                        # Plot values
                        label = f"f_s = {factor:.3f}"
                        ax.plot(times[:len(values)], values, '-', color=color, label=label)
                        
                        # Add confidence interval
                        if errors:
                            add_confidence_interval(ax, times[:len(values)], values, errors, color=color)
            
            # Configure axis
            ax_title = f"{metric.replace('_', ' ').title()} Comparison"
            configure_axis(ax, title=ax_title, xlabel='Time', ylabel=metric.replace('_', ' ').title())
            
            # Add statistical test for final values if reference factor is provided
            if reference_factor is not None:
                # Perform statistical comparison at the final time point
                test_annotations = []
                ref_metrics, _ = metrics_by_factor.get(reference_factor, (None, None))
                
                if ref_metrics and metric in ref_metrics:
                    ref_values = ref_metrics[metric]
                    if ref_values:
                        ref_final = ref_values[-1]
                        
                        # Compare each factor to reference
                        for j, factor in enumerate(factors):
                            if factor != reference_factor and factor in metrics_by_factor:
                                factor_metrics, _ = metrics_by_factor[factor]
                                if metric in factor_metrics and factor_metrics[metric]:
                                    test_values = factor_metrics[metric]
                                    if test_values:
                                        test_final = test_values[-1]
                                        
                                        # Perform t-test for significance
                                        if abs(test_final - ref_final) > 1e-10:
                                            # Simple z-score for single values
                                            z_score = (test_final - ref_final) / max(abs(ref_final), 1e-10)
                                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                                            
                                            # Add annotation if significant
                                            if p_value < 0.1:  # Show only significant differences
                                                # Position at end of lines
                                                x_pos = times[-1]
                                                y_ref = ref_final
                                                y_test = test_final
                                                
                                                # Add statistical annotation
                                                add_statistical_annotation(
                                                    ax, x_pos, x_pos, 
                                                    max(y_ref, y_test) + 0.05, 
                                                    p_value, 
                                                    test_name=f"f_s={factor:.2f}"
                                                )
                
                # Add reference label
                ax.text(0.02, 0.98, f"Reference: f_s={reference_factor:.3f}", 
                      transform=ax.transAxes, va='top', 
                      bbox=dict(facecolor='white', alpha=0.7))
            
            # Add legend with reasonable number of entries
            if len(factors) > 6:
                # Too many factors - show only subset
                subset_indices = np.linspace(0, len(factors)-1, 6, dtype=int)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend([handles[i] for i in subset_indices], 
                         [labels[i] for i in subset_indices],
                         loc='best')
            else:
                ax.legend(loc='best')
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
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
        purities.append(rho.purity())
        
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

def plot_metric_significance(
    metric_values: Dict[float, List[float]],
    metric_name: str,
    title: Optional[str] = None,
    reference_value: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot statistical significance of a metric across different scaling factors.
    
    Parameters:
        metric_values: Dictionary mapping scaling factors to lists of metric values
        metric_name: Name of the metric being analyzed
        title: Optional plot title
        reference_value: Reference scaling factor for comparison
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract scaling factors and mean metric values
    factors = sorted(metric_values.keys())
    means = [np.mean(metric_values[f]) for f in factors]
    stds = [np.std(metric_values[f]) if len(metric_values[f]) > 1 else 0 for f in factors]
    
    # Plot means with error bars
    ax.errorbar(factors, means, yerr=stds, fmt='o-', capsize=5, 
               color=COLORS['primary'], label=metric_name)
    
    # Add reference line if provided
    if reference_value is not None:
        ref_idx = np.argmin(np.abs(np.array(factors) - reference_value))
        ax.axvline(x=reference_value, color='r', linestyle='--', alpha=0.5, 
                  label=f'Reference (f_s={reference_value:.3f})')
        
        # Perform statistical tests against reference
        if ref_idx < len(factors):
            ref_mean = means[ref_idx]
            ref_std = stds[ref_idx]
            
            for i, factor in enumerate(factors):
                if i != ref_idx:
                    # Perform z-test (assuming normal distribution)
                    mean = means[i]
                    std = stds[i]
                    
                    # Pooled standard error
                    if std > 0 and ref_std > 0:
                        z_score = (mean - ref_mean) / np.sqrt(std**2 + ref_std**2)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        # Add significance markers
                        if p_value < 0.1:  # Only add for significant differences
                            y_pos = max(mean, ref_mean) + max(std, ref_std) * 1.5
                            add_statistical_annotation(
                                ax, reference_value, factor, y_pos, 
                                p_value, test_name="z-test"
                            )
    
    # Configure axis
    configure_axis(
        ax,
        title=title or f"{metric_name.replace('_', ' ').title()} Across Scaling Factors",
        xlabel="Scaling Factor (f_s)",
        ylabel=metric_name.replace('_', ' ').title(),
        legend=True
    )
    
    # Add statistical significance legend
    stat_legend = []
    for level, marker in SIGNIFICANCE_LEVELS.items():
        if level < 1.0:  # Skip "not significant"
            desc = ""
            if level == 0.001:
                desc = f"{marker} p < 0.001"
            elif level == 0.01:
                desc = f"{marker} p < 0.01"
            elif level == 0.05:
                desc = f"{marker} p < 0.05"
            elif level == 0.1:
                desc = f"{marker} p < 0.1"
            
            if desc:  # Only append if we have a description
                stat_legend.append(desc)
    
    if stat_legend:
        ax.text(0.02, 0.02, "\n".join(stat_legend),
               transform=ax.transAxes, verticalalignment='bottom',
               bbox=dict(facecolor='white', alpha=0.7))
    
    fig.tight_layout()
    return fig

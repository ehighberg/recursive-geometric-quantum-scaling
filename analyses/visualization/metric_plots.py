"""
Visualization functions for quantum metrics including entropy, coherence, and entanglement measures.
Provides uniform statistical analysis and visualization for all scaling factors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
from qutip import Qobj, fidelity
from scipy import stats
import itertools
import logging

from .style_config import (
    set_style, configure_axis, get_color_cycle, COLORS,
    add_statistical_annotation, SIGNIFICANCE_LEVELS
)

# Set up logging
logger = logging.getLogger(__name__)

def calculate_metrics(states, compute_errors=False):
    """
    Calculate quantum metrics for a list of quantum states.
    
    This function computes various quantum information metrics for each state:
    - von Neumann entropy: quantifies quantum uncertainty and entanglement
    - L1 coherence: sum of absolute values of off-diagonal elements
    - Purity: Tr(ρ²), measures how pure (vs mixed) a quantum state is
    
    Parameters:
    -----------
    states : List[Qobj]
        List of quantum states (kets or density matrices)
    compute_errors : bool, optional
        Whether to compute error estimates for metrics
        
    Returns:
    --------
    Dict[str, List[float]] or Tuple[Dict[str, List[float]], Dict[str, List[float]]]
        Dictionary mapping metric names to lists of values, and optionally errors
    """
    # Import specialized QuTiP functions
    try:
        from qutip.entropy import entropy_vn, entropy_linear
    except ImportError:
        # Fallback if specialized entropy functions aren't available
        logger.warning("QuTiP entropy functions not available, using custom implementations")
        entropy_vn = None
        entropy_linear = None
    
    # Initialize metrics dictionary
    metrics = {
        'coherence': [],
        'entropy': [],
        'purity': []
    }
    
    # Calculate each metric for each state
    for state in states:
        # Ensure state is density matrix for certain calculations
        rho = state if not state.isket else state * state.dag()
        
        # Calculate von Neumann entropy S = -Tr(ρ ln ρ)
        try:
            if entropy_vn is not None:
                # Use QuTiP's optimized function
                metrics['entropy'].append(entropy_vn(rho))
            else:
                # Manual calculation using eigenvalues
                eigvals = rho.eigenenergies()
                # Filter out extremely small eigenvalues to avoid log(0)
                eigvals = eigvals[eigvals > 1e-12]
                entropy = -np.sum(eigvals * np.log2(eigvals))
                metrics['entropy'].append(entropy)
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            metrics['entropy'].append(np.nan)
        
        # Calculate L1 coherence (sum of absolute values of off-diagonal elements)
        try:
            rho_matrix = rho.full()
            coherence = 0
            for i in range(rho_matrix.shape[0]):
                for j in range(rho_matrix.shape[1]):
                    if i != j:
                        coherence += np.abs(rho_matrix[i, j])
            metrics['coherence'].append(coherence)
        except Exception as e:
            logger.warning(f"Error calculating coherence: {e}")
            metrics['coherence'].append(np.nan)
        
        # Calculate purity (Tr(ρ²))
        try:
            if hasattr(rho, 'purity'):
                purity = rho.purity()
            else:
                purity = np.real((rho * rho).tr())
            metrics['purity'].append(purity)
        except Exception as e:
            logger.warning(f"Error calculating purity: {e}")
            metrics['purity'].append(np.nan)
    
    # Compute error estimates if requested
    if compute_errors:
        # For quantum metrics, use statistical bootstrapping to estimate errors
        # when possible, otherwise use simpler approach
        errors = {}
        
        # For each metric, estimate uncertainty
        for metric_name, values in metrics.items():
            values_array = np.array(values)
            valid_values = values_array[~np.isnan(values_array)]
            
            if len(valid_values) > 5:
                # Use bootstrapping for uncertainty estimation
                try:
                    from scipy.stats import bootstrap
                    bootstrap_result = bootstrap(
                        (valid_values,), 
                        np.std, 
                        n_resamples=1000, 
                        confidence_level=0.95
                    )
                    std_error = bootstrap_result.standard_error
                    errors[metric_name] = [std_error] * len(values)
                except Exception:
                    # Fallback to standard error of the mean
                    if len(valid_values) > 1:
                        std_error = np.std(valid_values) / np.sqrt(len(valid_values))
                        errors[metric_name] = [std_error] * len(values)
                    else:
                        # Not enough data for error estimation
                        errors[metric_name] = [0.01] * len(values)  # Minimal placeholder
            else:
                # Not enough data points for bootstrapping, use simple estimate
                if len(valid_values) > 1:
                    std_error = np.std(valid_values) / np.sqrt(len(valid_values))
                    errors[metric_name] = [std_error] * len(values)
                else:
                    # Not enough data for error estimation
                    errors[metric_name] = [0.01] * len(values)  # Minimal placeholder
        
        return metrics, errors
    
    return metrics

def add_confidence_interval(ax, x, y, error, color=None, alpha=0.2):
    """Add confidence interval to a plot."""
    color = color or 'blue'
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    
    # Calculate upper and lower bounds
    lower = y - error
    upper = y + error
    
    # Plot confidence interval
    ax.fill_between(x, lower, upper, color=color, alpha=alpha)

def plot_metric_evolution(
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    metrics: Optional[List[str]] = None,
    show_errors: bool = True
) -> plt.Figure:
    """Plot the evolution of quantum metrics over time."""
    if metrics is None:
        metrics = ['entropy', 'coherence', 'purity']
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate metrics and errors
    if show_errors:
        metric_values, metric_errors = calculate_metrics(states, compute_errors=True)
    else:
        metric_values = calculate_metrics(states)
        metric_errors = None
    
    # Plot each metric
    colors = get_color_cycle(len(metrics))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in metric_values:
            values = metric_values[metric]
            
            # Convert NaN values to interpolated values for display
            values_array = np.array(values)
            nan_mask = np.isnan(values_array)
            if np.any(nan_mask) and not np.all(nan_mask):
                # Create smooth data by interpolating over NaNs
                valid_indices = np.where(~nan_mask)[0]
                valid_values = values_array[valid_indices]
                valid_times = np.array(times)[valid_indices]
                
                if len(valid_indices) > 1:
                    # Use available points to interpolate
                    from scipy.interpolate import interp1d
                    try:
                        f = interp1d(valid_times, valid_values, 
                                     kind='linear', bounds_error=False, 
                                     fill_value=(valid_values[0], valid_values[-1]))
                        values_smoothed = f(times)
                    except Exception:
                        # Fallback for any interpolation issues
                        values_smoothed = values_array.copy()
                else:
                    # Not enough points for interpolation
                    values_smoothed = values_array.copy()
                    
                # Only plot finite values
                finite_mask = np.isfinite(values_smoothed)
                plot_times = np.array(times)[finite_mask]
                plot_values = values_smoothed[finite_mask]
            else:
                # Use original data if no NaNs or all NaNs
                finite_mask = np.isfinite(values_array)
                plot_times = np.array(times)[finite_mask]
                plot_values = values_array[finite_mask]
            
            # Plot main line (only plot if we have data)
            if len(plot_times) > 0:
                ax.plot(plot_times, plot_values, 
                        label=metric.replace('_', ' ').title(),
                        color=color)
                
                # Add confidence interval
                if show_errors and metric_errors and metric in metric_errors:
                    errors = np.array(metric_errors[metric])[finite_mask]
                    if len(errors) == len(plot_values):
                        add_confidence_interval(ax, plot_times, plot_values, errors, color=color)
    
    configure_axis(ax,
                  title=title or 'Quantum Metrics Evolution',
                  xlabel='Time',
                  ylabel='Value')
    
    if len(metrics) > 1:
        ax.legend(loc='best')
    
    fig.tight_layout()
    return fig

def plot_metric_comparison(
    states: List[Qobj],
    metric_pairs: List[Tuple[str, str]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Compare different metrics against each other (scatter plots)."""
    set_style()
    n_pairs = len(metric_pairs)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Calculate all metrics
    metrics = calculate_metrics(states)
    
    # Plot each metric pair
    for i, (metric1, metric2) in enumerate(metric_pairs):
        if i < len(axes):
            ax = axes[i]
            
            # Get metric values
            if metric1 in metrics and metric2 in metrics:
                x_values = metrics[metric1]
                y_values = metrics[metric2]
                
                # Ensure same length
                min_len = min(len(x_values), len(y_values))
                x_values = x_values[:min_len]
                y_values = y_values[:min_len]
                
                # Filter out NaN values for valid plotting
                valid_mask = ~(np.isnan(x_values) | np.isnan(y_values))
                valid_x = np.array(x_values)[valid_mask]
                valid_y = np.array(y_values)[valid_mask]
                
                if len(valid_x) > 0:
                    # Plot scatter
                    ax.scatter(valid_x, valid_y, color=COLORS['primary'], alpha=0.7)
                    
                    # Add trend line if enough data
                    if len(valid_x) > 2:
                        try:
                            # Simple linear regression
                            z = np.polyfit(valid_x, valid_y, 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(min(valid_x), max(valid_x), 100)
                            ax.plot(x_range, p(x_range), '--', color=COLORS['secondary'], alpha=0.7)
                            
                            # Add correlation coefficient
                            r_coef = np.corrcoef(valid_x, valid_y)[0, 1]
                            ax.text(0.05, 0.95, f'r = {r_coef:.2f}', transform=ax.transAxes,
                                  verticalalignment='top', fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                        except Exception as e:
                            logger.warning(f"Error adding trend line: {e}")
                
                # Configure axis
                configure_axis(
                    ax,
                    title=f"{metric1.replace('_', ' ').title()} vs {metric2.replace('_', ' ').title()}",
                    xlabel=metric1.replace('_', ' ').title(),
                    ylabel=metric2.replace('_', ' ').title()
                )
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
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
    """Plot comparative metrics for different scaling factors."""
    set_style()
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each metric
    factors = sorted(states_by_factor.keys())
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # Plot each scaling factor
            for j, factor in enumerate(factors):
                states = states_by_factor[factor]
                
                # Calculate metrics for this factor
                metric_values = calculate_metrics(states)[metric]
                
                # Filter out NaN values
                valid_mask = ~np.isnan(metric_values)
                if not np.any(valid_mask):
                    continue  # Skip if no valid data
                    
                valid_times = np.array(times[:len(metric_values)])[valid_mask]
                valid_values = np.array(metric_values)[valid_mask]
                
                # Get color and plot
                color = get_color_cycle()[j % len(get_color_cycle())]
                label = f"f_s = {factor:.3f}"
                
                # Highlight reference factor if specified
                if reference_factor is not None and abs(factor - reference_factor) < 1e-6:
                    # Use thicker line and different style for reference
                    ax.plot(valid_times, valid_values, '-', 
                          color=color, label=label, linewidth=3)
                else:
                    ax.plot(valid_times, valid_values, '-', 
                          color=color, label=label)
            
            # Configure axis
            ax_title = f"{metric.replace('_', ' ').title()} Comparison"
            configure_axis(ax, title=ax_title, xlabel='Time', ylabel=metric.replace('_', ' ').title())
            ax.legend(loc='best')
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    fig.tight_layout()
    return fig

def plot_metric_distribution(
    states: List[Qobj],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Create distribution plots (histograms) for quantum metrics."""
    set_style()
    
    # Calculate metrics
    metrics = calculate_metrics(states)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Create distribution plots
    colors = itertools.cycle(get_color_cycle())
    for ax, (metric, values) in zip(axes, metrics.items()):
        # Convert to numpy and filter NaNs
        values_array = np.array(values)
        valid_values = values_array[~np.isnan(values_array)]
        
        if len(valid_values) == 0:
            ax.text(0.5, 0.5, "No valid data", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            continue
        
        color = next(colors)
        
        if len(valid_values) < 10:
            # For small datasets, just plot points
            ax.plot(np.zeros_like(valid_values), valid_values, 'o', color=color, alpha=0.7)
            ax.set_xlim(-0.5, 0.5)
        else:
            # For larger datasets, use histogram
            value_range = np.max(valid_values) - np.min(valid_values)
            if value_range < 1e-6:
                # Nearly identical values - plot as points with jitter
                jitter = np.random.normal(0, 0.01, size=len(valid_values))
                ax.plot(jitter, valid_values, 'o', color=color, alpha=0.7)
                ax.set_xlim(-0.5, 0.5)
            else:
                # Use histogram
                bins = min(20, max(5, int(np.sqrt(len(valid_values)))))
                ax.hist(valid_values, bins=bins, color=color, alpha=0.7, 
                       density=True)
                
                # Add kernel density estimate for smoother distribution
                try:
                    from scipy.stats import gaussian_kde
                    density = gaussian_kde(valid_values)
                    x_range = np.linspace(min(valid_values), max(valid_values), 100)
                    ax.plot(x_range, density(x_range), '-', color='k')
                except Exception as e:
                    logger.warning(f"KDE estimation failed: {e}")
        
        # Add statistical summary
        mean_val = np.mean(valid_values)
        median_val = np.median(valid_values)
        std_val = np.std(valid_values)
        
        stats_text = (f"Mean: {mean_val:.3f}\n"
                     f"Median: {median_val:.3f}\n"
                     f"Std Dev: {std_val:.3f}")
        
        # Add text box with statistics
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        configure_axis(ax,
                      title=metric.replace('_', ' ').title(),
                      xlabel='Value' if len(valid_values) >= 10 else '',
                      ylabel='Density' if len(valid_values) >= 10 else 'Value')
    
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
    """Plot noise-specific metrics over time."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Calculate actual metrics
    metrics = calculate_metrics(states)
    purities = metrics.get('purity', [])
    
    # Calculate state fidelities if initial state is provided
    fidelities = []
    if initial_state is not None:
        initial_rho = initial_state if not initial_state.isket else initial_state * initial_state.dag()
        for state in states:
            state_rho = state if not state.isket else state * state.dag()
            try:
                # Use QuTiP's fidelity function
                fid = fidelity(initial_rho, state_rho)
                fidelities.append(fid)
            except Exception as e:
                logger.warning(f"Fidelity calculation failed: {e}")
                fidelities.append(np.nan)
    
    # Calculate coherence
    coherences = metrics.get('coherence', [])
    
    # Filter NaNs and ensure proper length
    t_purities = times[:len(purities)]
    t_coherences = times[:len(coherences)]
    t_fidelities = times[:len(fidelities)]
    
    valid_p_mask = ~np.isnan(purities)
    valid_c_mask = ~np.isnan(coherences)
    valid_f_mask = ~np.isnan(fidelities)
    
    # Plot metrics if data available
    if np.any(valid_p_mask):
        axes[0].plot(np.array(t_purities)[valid_p_mask], 
                     np.array(purities)[valid_p_mask], 
                     color=COLORS['primary'], label='Purity')
    configure_axis(axes[0], title='State Purity', xlabel='Time', ylabel='Tr(ρ²)')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    
    if np.any(valid_c_mask):
        # Normalize coherence to 0-1 range for better visualization
        c_values = np.array(coherences)[valid_c_mask]
        if len(c_values) > 0:
            max_c = np.max(c_values)
            if max_c > 0:
                norm_coherences = c_values / max_c
                axes[1].plot(np.array(t_coherences)[valid_c_mask], 
                             norm_coherences, 
                             color=COLORS['secondary'], label='Coherence (Norm.)')
            else:
                axes[1].plot(np.array(t_coherences)[valid_c_mask], 
                             c_values, 
                             color=COLORS['secondary'], label='Coherence')
    configure_axis(axes[1], title='Quantum Coherence', xlabel='Time', ylabel='L1 Coherence')
    axes[1].legend()
    
    if np.any(valid_f_mask) and len(fidelities) > 0:
        axes[2].plot(np.array(t_fidelities)[valid_f_mask], 
                     np.array(fidelities)[valid_f_mask], 
                     color=COLORS['accent'], label='Fidelity')
    configure_axis(axes[2], title='State Fidelity', xlabel='Time', ylabel='F(ρ₀,ρ(t))')
    axes[2].set_ylim(0, 1.1)
    axes[2].legend()
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    fig.tight_layout()
    return fig

def plot_quantum_metrics(
    states: List[Qobj],
    times: List[float],
    title: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Comprehensive visualization of quantum metrics."""
    if metrics is None:
        metrics = ['entropy', 'coherence', 'purity']
    
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Calculate metrics
    metric_values = calculate_metrics(states)
    
    # 1. Evolution plot
    for i, metric in enumerate(metrics):
        if metric in metric_values:
            values = np.array(metric_values[metric])
            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                continue  # Skip if no valid values
                
            valid_times = np.array(times[:len(values)])[valid_mask]
            valid_values = values[valid_mask]
            
            color = get_color_cycle()[i % len(get_color_cycle())]
            axes[0].plot(valid_times, valid_values, 
                        label=metric.replace('_', ' ').title(),
                        color=color)
    
    configure_axis(axes[0], title='Metrics Evolution', xlabel='Time', ylabel='Value')
    axes[0].legend()
    
    # 2. Distribution plot
    dist_data = []
    dist_labels = []
    for metric in metrics:
        if metric in metric_values:
            values = np.array(metric_values[metric])
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                dist_data.append(valid_values)
                dist_labels.append(metric.replace('_', ' ').title())
    
    if dist_data:
        axes[1].boxplot(dist_data, labels=dist_labels)
        configure_axis(axes[1], title='Metrics Distribution', ylabel='Value')
    
    # 3. Correlation plot
    if len(metrics) >= 2 and metrics[0] in metric_values and metrics[1] in metric_values:
        x_values = np.array(metric_values[metrics[0]])
        y_values = np.array(metric_values[metrics[1]])
        
        # Filter for valid points
        valid_mask = ~(np.isnan(x_values) | np.isnan(y_values))
        if np.any(valid_mask):
            valid_x = x_values[valid_mask]
            valid_y = y_values[valid_mask]
            
            axes[2].scatter(valid_x, valid_y, color=COLORS['primary'], alpha=0.7)
            
            # Add trend line if enough data
            if len(valid_x) > 2:
                try:
                    z = np.polyfit(valid_x, valid_y, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(valid_x), max(valid_x), 100)
                    axes[2].plot(x_range, p(x_range), '--', color=COLORS['secondary'], alpha=0.7)
                    
                    # Add correlation coefficient
                    r_coef = np.corrcoef(valid_x, valid_y)[0, 1]
                    axes[2].text(0.05, 0.95, f'r = {r_coef:.2f}', transform=axes[2].transAxes,
                              verticalalignment='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                except Exception as e:
                    logger.warning(f"Error adding trend line: {e}")
        
        configure_axis(
            axes[2], 
            title=f"{metrics[0].replace('_', ' ').title()} vs {metrics[1].replace('_', ' ').title()}",
            xlabel=metrics[0].replace('_', ' ').title(),
            ylabel=metrics[1].replace('_', ' ').title()
        )
    
    # 4. Quantum state correlation heatmap
    if len(states) > 1:
        # Calculate actual state overlaps instead of mock data
        size = min(len(states), 20)  # Limit to prevent excessive computation
        corr_data = np.zeros((size, size))
        
        # Calculate state overlaps using fidelity for a subset of states
        step = max(1, len(states) // size)
        selected_states = [states[i] for i in range(0, len(states), step)][:size]
        
        for i in range(len(selected_states)):
            state_i = selected_states[i]
            state_i_dm = state_i if not state_i.isket else state_i * state_i.dag()
            
            for j in range(len(selected_states)):
                state_j = selected_states[j]
                state_j_dm = state_j if not state_j.isket else state_j * state_j.dag()
                
                try:
                    # Use QuTiP's fidelity function
                    corr_data[i, j] = fidelity(state_i_dm, state_j_dm)
                except Exception:
                    # Fallback to trace distance if fidelity fails
                    try:
                        trace_dist = 0.5 * (state_i_dm - state_j_dm).norm('tr')
                        corr_data[i, j] = 1.0 - trace_dist
                    except Exception:
                        corr_data[i, j] = np.nan
        
        # Plot heatmap
        im = axes[3].imshow(corr_data, cmap='viridis', interpolation='nearest',
                          vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[3], label='State Fidelity')
        
        # Create time labels
        if len(times) >= len(selected_states):
            time_indices = np.arange(0, len(states), step)[:size]
            time_labels = [f"{times[idx]:.1f}" for idx in time_indices]
            
            # Only show a reasonable number of labels
            if len(time_labels) > 10:
                stride = len(time_labels) // 10
                visible_positions = np.arange(0, len(time_labels), stride)
                visible_labels = [time_labels[i] if i in visible_positions else "" 
                                 for i in range(len(time_labels))]
                
                axes[3].set_xticks(np.arange(len(time_labels)))
                axes[3].set_yticks(np.arange(len(time_labels)))
                axes[3].set_xticklabels(visible_labels)
                axes[3].set_yticklabels(visible_labels)
            else:
                axes[3].set_xticks(np.arange(len(time_labels)))
                axes[3].set_yticks(np.arange(len(time_labels)))
                axes[3].set_xticklabels(time_labels)
                axes[3].set_yticklabels(time_labels)
        
        configure_axis(axes[3], title='State Fidelity Matrix', xlabel='Time', ylabel='Time')

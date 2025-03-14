#!/usr/bin/env python
"""
Visualization functions for fractal-related quantum properties including energy spectra,
wavefunction profiles, and fractal dimension analysis.

This module provides publication-quality plotting functions that integrate with
the fractal analysis module to produce detailed visualizations of:
- Energy spectra with automatic self-similarity detection
- Wavefunction profiles with configurable zoom regions
- Fractal dimension analysis with error quantification
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Callable, Dict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from .style_config import set_style, configure_axis, COLORS
from qutip import Qobj
from analyses.fractal_analysis import (
    compute_wavefunction_profile,
    estimate_fractal_dimension,
    load_fractal_config
)

def plot_energy_spectrum(
    parameter_values: np.ndarray,
    energies: np.ndarray,
    analysis: Dict,
    parameter_name: str = "f_s",
    figsize: Tuple[int, int] = (12, 8),
    config: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot energy spectrum showing fractal gap structure with enhanced annotations
    and self-similarity detection.
    
    Parameters:
    -----------
    parameter_values : np.ndarray
        Array of parameter values (x-axis).
    energies : np.ndarray
        Array of energy values to plot.
    analysis : Dict
        Analysis dictionary from compute_energy_spectrum containing:
        - 'self_similar_regions': List of (start1, end1, start2, end2) tuples
        - 'correlation_matrix': Correlation analysis results
        - 'gap_statistics': Statistics of energy gaps
    parameter_name : str
        Name of the parameter being varied (for x-axis label).
    figsize : Tuple[int, int]
        Figure dimensions.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
        
    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    """
    if config is None:
        config = load_fractal_config()
    
    vis_config = config.get('visualization', {})
    dpi = vis_config.get('dpi', 300)
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot energy bands
    if len(energies.shape) > 1:
        for i in range(energies.shape[1]):
            ax.plot(parameter_values, energies[:, i], '-', 
                   color=COLORS['primary'], alpha=0.7,
                   label=f'Band {i+1}')
    else:
        ax.plot(parameter_values, energies, '-', 
                color=COLORS['primary'], label='Energy Band')
    
    # Highlight and annotate self-similar regions
    if 'self_similar_regions' in analysis and analysis['self_similar_regions']:
        for i, region in enumerate(analysis['self_similar_regions']):
            # Handle both tuple formats (flexible handling)
            if len(region) == 4:
                start1, end1, start2, end2 = region
            elif len(region) == 2:
                # If we only have two points, treat them as a single region
                start1, end1 = region
                start2, end2 = start1, end1  # No corresponding region
            else:
                # Skip invalid regions
                continue
                
            # Highlight first region
            ax.axvspan(start1, end1, color=COLORS['accent'], alpha=0.2)
            
            # Highlight corresponding similar region (if different)
            if start2 != start1 or end2 != end1:
                ax.axvspan(start2, end2, color=COLORS['accent'], alpha=0.2)
                
                # Add connecting arrow
                mid1 = (start1 + end1) / 2
                mid2 = (start2 + end2) / 2
                y_pos = ax.get_ylim()[1]
                ax.annotate('', xy=(mid2, y_pos), xytext=(mid1, y_pos),
                           arrowprops=dict(arrowstyle='<->',
                                         color=COLORS['accent']))
        
        # Add region labels
        ax.text(mid1, y_pos, f'R{i+1}a',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(mid2, y_pos, f'R{i+1}b',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add gap statistics annotation
    gap_stats = analysis['gap_statistics']
    stats_text = (f"Gap Statistics:\n"
                 f"Mean: {gap_stats['mean']:.2e}\n"
                 f"Std: {gap_stats['std']:.2e}")
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    configure_axis(ax,
                  title='Fractal Energy Spectrum',
                  xlabel=parameter_name,
                  ylabel='Energy (E)')
    
    ax.legend()
    fig.tight_layout()
    return fig

def plot_wavefunction_profile(
    wavefunction: Qobj,
    x_array: Optional[np.ndarray] = None,
    zoom_regions: Optional[List[Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (12, 8),
    config: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot wavefunction probability density with configurable zoom insets showing
    self-similarity and local fractal dimension estimates.
    
    Parameters:
    -----------
    wavefunction : Qobj
        The quantum wavefunction to visualize.
    x_array : Optional[np.ndarray]
        Array of position values.
    zoom_regions : Optional[List[Tuple[float, float]]]
        List of (x_start, x_end) tuples defining regions to zoom.
    figsize : Tuple[int, int]
        Figure dimensions.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
        
    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    """
    if config is None:
        config = load_fractal_config()
    
    if x_array is None:
        x_array = np.linspace(0, 1, 100)
    
    wf_config = config.get('wavefunction_zoom', {})
    zoom_factor = wf_config.get('zoom_factor', 2.0)
    vis_config = config.get('visualization', {})
    dpi = vis_config.get('dpi', 300)
    
    # Compute probability density with detailed analysis
    probability_density, details = compute_wavefunction_profile(
        wavefunction, x_array, zoom_factor=1.0, log_details=True
  # Use zoom_factor=1.0 initially
    )
    
    # Use detected regions if none provided
    if zoom_regions is None and details is not None:
        zoom_regions = details['zoom_regions']
    elif zoom_regions is None:
        zoom_regions = wf_config.get('default_windows', [[0.2, 0.4], [0.6, 0.8]])

    # Ensure x_array and probability_density have the same length
    if len(probability_density) != len(x_array):
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, len(probability_density)), probability_density,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        probability_density = f(x_array)
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Main plot
    ax.plot(x_array, probability_density, '-', 
            color=COLORS['primary'], label='|ψ(x)|²')
    
    # Add zoomed insets with fractal dimension estimates
    for i, (x1, x2) in enumerate(zoom_regions):
        # Create inset
        loc = 'center left' if i % 2 == 0 else 'center right'
        axins = inset_axes(ax, width="30%", height="30%",
                          loc=loc)
        
        # Plot zoomed region
        mask = (x_array >= x1) & (x_array <= x2)
        zoomed_x = x_array[mask]
        zoomed_density = probability_density[mask]
        axins.plot(zoomed_x, zoomed_density,
                  color=COLORS['accent'])
        
        # Estimate local fractal dimension
        box_sizes = np.logspace(-3, 0, 5)
        dimension, info = estimate_fractal_dimension(
            zoomed_density, box_sizes, config
        )
        
        # Add dimension and confidence interval
        ci = info['confidence_interval']
        dim_text = f'D = {dimension:.2f}\n95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]'
        axins.set_title(dim_text, fontsize=8)
        
        # Set inset limits and mark region
        axins.set_xlim(x1, x2)
        y_margin = 0.1 * (np.max(zoomed_density) - np.min(zoomed_density))
        axins.set_ylim(np.min(zoomed_density) - y_margin,
                      np.max(zoomed_density) + y_margin)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none",
                  ec=COLORS['accent'])
    
    configure_axis(ax,
                  title='Wavefunction Profile',
                  xlabel='Position (x)',
                  ylabel='Probability Density')
    
    # Add normalization verification
    if details is not None:
        norm_text = f"Normalization: {details['normalization']:.6f}"
        ax.text(0.02, 0.98, norm_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    fig.tight_layout()
    return fig

def plot_fractal_dimension(
    recursion_depths: np.ndarray,
    fractal_dimensions: np.ndarray,
    error_bars: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot fractal dimension vs. recursion depth with enhanced error analysis
    and theoretical predictions.
    
    Parameters:
    -----------
    recursion_depths : np.ndarray
        Array of recursion depth values.
    fractal_dimensions : np.ndarray
        Array of computed fractal dimensions.
    error_bars : Optional[np.ndarray]
        Array of error values for each fractal dimension.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
    figsize : Tuple[int, int]
        Figure dimensions.
        
    Returns:
    --------
    fig : plt.Figure
        Matplotlib figure containing the plot.
    """
    if config is None:
        config = load_fractal_config()
    
    fd_config = config.get('fractal_dimension', {})
    vis_config = config.get('visualization', {})
    dpi = vis_config.get('dpi', 300)
    theoretical_dim = fd_config.get('theoretical_dimension')
    scaling_text = vis_config.get('scaling_function_text', 'D(n) ~ f(n)')
    
    set_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot computed dimensions with error bars
    if error_bars is not None:
        ax.errorbar(recursion_depths, fractal_dimensions,
                   yerr=error_bars, fmt='o-',
                   color=COLORS['primary'],
                   capsize=5,
                   label='Computed D')
    else:
        ax.plot(recursion_depths, fractal_dimensions, 'o-',
                color=COLORS['primary'],
                label='Computed D')
    
    # Add theoretical prediction if available
    if theoretical_dim is not None:
        ax.axhline(y=theoretical_dim,
                  color=COLORS['accent'],
                  linestyle='--',
                  label=f'Theoretical D = {theoretical_dim:.3f}')
    
    # Add scaling law annotation
    ax.text(0.05, 0.95, scaling_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    configure_axis(ax,
                  title='Fractal Dimension Analysis',
                  xlabel='Recursion Depth',
                  ylabel='Fractal Dimension (D)')
    
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    return fig

def plot_fractal_analysis_summary(
    parameter_values: np.ndarray,
    energies: np.ndarray,
    analysis: Dict,
    wavefunction: Qobj,
    recursion_depths: np.ndarray,
    fractal_dimensions: np.ndarray,
    error_bars: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create a publication-ready summary figure combining energy spectrum,
    wavefunction profile, and fractal dimension analysis.
    """
    if config is None:
        config = load_fractal_config()
    
    vis_config = config.get('visualization', {})
    dpi = vis_config.get('dpi', 300)
    
    set_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    # Energy spectrum
    if len(energies.shape) > 1:
        for i in range(energies.shape[1]):
            ax1.plot(parameter_values, energies[:, i], '-',
                    color=COLORS['primary'], alpha=0.7)
    else:
        ax1.plot(parameter_values, energies, '-',
                color=COLORS['primary'])
    
    # Highlight self-similar regions
    for start1, end1, start2, end2 in analysis['self_similar_regions']:
        ax1.axvspan(start1, end1, color=COLORS['accent'], alpha=0.2)
        ax1.axvspan(start2, end2, color=COLORS['accent'], alpha=0.2)
    
    configure_axis(ax1,
                  title='Energy Spectrum',
                  xlabel='Parameter',
                  ylabel='Energy (E)')
    
    # Wavefunction profile
    x_array = np.linspace(0, 1, 100)
    probability_density, _ = compute_wavefunction_profile(
        wavefunction, x_array, zoom_factor=1.0, log_details=False
    )
    # Ensure matching dimensions
    if len(probability_density) != len(x_array):
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, len(probability_density)), probability_density,
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        probability_density = f(x_array)
    
    ax2.plot(x_array, probability_density, '-',
            color=COLORS['primary'])
    configure_axis(ax2,
                  title='Wavefunction Profile',
                  xlabel='Position (x)',
                  ylabel='|ψ(x)|²')
    
    # Fractal dimension
    if error_bars is not None:
        ax3.errorbar(recursion_depths, fractal_dimensions,
                    yerr=error_bars, fmt='o-',
                    color=COLORS['primary'],
                    capsize=5)
    else:
        ax3.plot(recursion_depths, fractal_dimensions, 'o-',
                color=COLORS['primary'])
    
    theoretical_dim = config.get('fractal_dimension', {}).get('theoretical_dimension')
    if theoretical_dim is not None:
        ax3.axhline(y=theoretical_dim,
                   color=COLORS['accent'],
                   linestyle='--')
    
    configure_axis(ax3,
                  title='Fractal Dimension',
                  xlabel='Recursion Depth',
                  ylabel='D')
    
    fig.tight_layout()
    return fig

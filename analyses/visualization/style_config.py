"""
Style configuration for quantum visualization components.
Provides consistent styling across all plots with unbiased treatment of scaling factors.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

# Enhanced color schemes with more neutrality
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'accent': '#2ca02c',       # Green
    'highlight': '#d62728',    # Red
    'purple': '#9467bd',       # Purple
    'brown': '#8c564b',        # Brown
    'pink': '#e377c2',         # Pink
    'gray': '#7f7f7f',         # Gray
    'olive': '#bcbd22',        # Olive
    'cyan': '#17becf',         # Cyan
    'background': '#ffffff',
    'text': '#333333',
    'grid': '#cccccc',
    'error_bars': '#888888',
    'confidence': 'lightgray',
    'significance': '#6b8e23',  # Olive green for significance indicators
}

# Colormaps for different plot types
CMAPS = {
    'sequential': 'viridis',        # For continuous data (prefer over jet)
    'diverging': 'RdBu_r',          # For data with natural midpoint
    'categorical': 'tab10',         # For discrete categories
    'density': 'plasma',            # For density plots
    'phase': 'twilight_shifted',    # For phase plots (circular)
    'confidence': 'Blues',          # For confidence regions
    'heatmap': 'viridis',           # For correlation matrices
}

# Unified plot styles with improved defaults for scientific visualization
PLOT_STYLE = {
    # Figure properties
    'figure.figsize': (10, 6),
    'figure.dpi': 150,                # Higher DPI for better quality
    'figure.facecolor': '#ffffff',
    'savefig.dpi': 300,               # Publication quality
    'savefig.facecolor': '#ffffff',
    'savefig.bbox': 'tight',
    
    # Fonts and text
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'lightgray',
    
    # Grid and ticks
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': COLORS['grid'],
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    
    # Color and styling
    'axes.facecolor': COLORS['background'],
    'axes.edgecolor': COLORS['text'],
    'axes.linewidth': 1.0,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.prop_cycle': cycler('color', [
        COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
        COLORS['highlight'], COLORS['purple'], COLORS['brown'],
        COLORS['pink'], COLORS['gray'], COLORS['olive'], COLORS['cyan']
    ]),
    
    # Error bars
    'errorbar.capsize': 3,
}

# Statistical significance levels with corresponding markers
SIGNIFICANCE_LEVELS = {
    0.001: '***',  # p < 0.001
    0.01: '**',    # p < 0.01
    0.05: '*',     # p < 0.05
    0.1: '†',      # p < 0.1 (marginally significant)
    1.0: 'n.s.'    # not significant
}

# Default colormaps
DENSITY_CMAP = CMAPS['diverging']
SEQUENTIAL_CMAP = CMAPS['sequential']

def set_style(style='default'):
    """
    Apply style settings with enhanced options.
    
    Parameters:
    -----------
    style : str
        Style to apply: 'default', 'scientific', 'presentation', or 'grayscale'
    """
    # Reset to default matplotlib style
    plt.style.use('default')
    
    # Apply our base style
    plt.rcParams.update(PLOT_STYLE)
    
    # Apply additional style-specific settings
    if style == 'scientific':
        # Publication-ready style with more minimal design
        plt.rcParams.update({
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
    elif style == 'presentation':
        # Bold style for presentations
        plt.rcParams.update({
            'lines.linewidth': 3,
            'axes.linewidth': 2,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
    elif style == 'grayscale':
        # Grayscale style for publications
        plt.style.use('grayscale')
        plt.rcParams.update(PLOT_STYLE)

def get_color_cycle(n=None):
    """
    Return color cycle for plots with a specified number of colors.
    
    Parameters:
    -----------
    n : int or None
        Number of colors to return. If None, returns all available colors.
        
    Returns:
    --------
    list : List of hex color codes
    """
    # Full color list
    colors = [
        COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
        COLORS['highlight'], COLORS['purple'], COLORS['brown'],
        COLORS['pink'], COLORS['gray'], COLORS['olive'], COLORS['cyan']
    ]
    
    if n is None:
        return colors
    
    # If more colors requested than available, cycle through
    if n <= len(colors):
        return colors[:n]
    else:
        return colors + [colors[i % len(colors)] for i in range(len(colors), n)]

def format_complex_number(z, precision=2):
    """
    Format complex number for plot labels.
    
    Parameters:
    -----------
    z : complex
        Complex number to format
    precision : int
        Number of decimal places to include
        
    Returns:
    --------
    str : Formatted complex number string
    """
    if not isinstance(z, complex):
        return f"{z:.{precision}f}"
    
    if z.imag == 0:
        return f"{z.real:.{precision}f}"
    elif z.real == 0:
        return f"{z.imag:.{precision}f}i"
    else:
        sign = '+' if z.imag >= 0 else ''
        return f"{z.real:.{precision}f}{sign}{z.imag:.{precision}f}i"

def configure_axis(ax, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, 
                  grid=True, legend=True, legend_loc='best'):
    """
    Configure axis with consistent styling and enhanced options.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to configure
    title : str or None
        Title for the plot
    xlabel : str or None
        Label for x axis
    ylabel : str or None
        Label for y axis
    xlim : tuple or None
        Limits for x axis (min, max)
    ylim : tuple or None
        Limits for y axis (min, max)
    grid : bool
        Whether to show grid
    legend : bool
        Whether to show legend
    legend_loc : str
        Location for legend
        
    Returns:
    --------
    ax : The configured matplotlib axis
    """
    if isinstance(title, (str, type(None))):
        if title:
            ax.set_title(title, pad=20)
    
    if isinstance(xlabel, (str, type(None))):
        if xlabel:
            ax.set_xlabel(xlabel)
    
    if isinstance(ylabel, (str, type(None))):
        if ylabel:
            ax.set_ylabel(ylabel)
    
    if xlim is not None:
        ax.set_xlim(xlim)
        
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.grid(grid, alpha=0.3, color=COLORS['grid'])
    ax.set_facecolor(COLORS['background'])
    
    if legend and len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.9)
    
    return ax

def add_statistical_annotation(ax, x1, x2, y, p_value, test_name=None, line_height=0.02):
    """
    Add statistical significance annotation between two points on a plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to annotate
    x1, x2 : float
        x-coordinates of the two points to connect
    y : float
        y-coordinate for the annotation
    p_value : float
        p-value from statistical test
    test_name : str or None
        Name of statistical test used
    line_height : float
        Height of the connecting line in axis coordinates
        
    Returns:
    --------
    annotation : The created annotation
    """
    # Determine significance marker
    if p_value < 0.001:
        marker = SIGNIFICANCE_LEVELS[0.001]
    elif p_value < 0.01:
        marker = SIGNIFICANCE_LEVELS[0.01]
    elif p_value < 0.05:
        marker = SIGNIFICANCE_LEVELS[0.05]
    elif p_value < 0.1:
        marker = SIGNIFICANCE_LEVELS[0.1]
    else:
        marker = SIGNIFICANCE_LEVELS[1.0]
    
    # Calculate annotation position
    yrange = ax.get_ylim()
    y_span = yrange[1] - yrange[0]
    
    # Create connecting line with bar
    ax.plot([x1, x1, x2, x2], [y, y + line_height * y_span, y + line_height * y_span, y], 
            color=COLORS['text'], linewidth=1)
    
    # Create annotation text
    if test_name:
        text = f"{marker} (p={p_value:.3f}, {test_name})"
    else:
        text = f"{marker} (p={p_value:.3f})"
    
    # Add annotation text
    ann = ax.annotate(text, xy=((x1 + x2) / 2, y + line_height * y_span * 1.1), 
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', 
                     color=COLORS['significance'],
                     fontweight='bold')
    
    return ann

def add_confidence_interval(ax, x, y, error, color=None, alpha=0.2):
    """
    Add a shaded confidence interval to a line plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add the confidence interval to
    x : array-like
        x-coordinates of the data points
    y : array-like
        y-coordinates of the data points
    error : array-like or float
        Error values for the confidence interval
    color : str or None
        Color for the confidence interval. If None, matches the line color.
    alpha : float
        Transparency of the confidence interval
        
    Returns:
    --------
    collection : The polygon collection representing the confidence interval
    """
    if color is None:
        # Try to get color from the last line added to the plot
        if ax.lines:
            color = ax.lines[-1].get_color()
        else:
            color = COLORS['primary']
    
    # Convert error to array if it's a scalar
    if np.isscalar(error):
        error = np.ones_like(y) * error
        
    # Create upper and lower bounds
    lower = y - error
    upper = y + error
    
    # Create and return the confidence interval
    collection = ax.fill_between(x, lower, upper, color=color, alpha=alpha)
    
    return collection

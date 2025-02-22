"""
Style configuration for quantum visualization components.
Provides consistent styling across all plots.
"""

import matplotlib.pyplot as plt

# Color schemes
COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'accent': '#2ca02c',  # Green
    'highlight': '#d62728',  # Red
    'background': '#ffffff',
    'text': '#333333',
}

# Plot styles
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
}

# Colormap for density matrices and heatmaps
DENSITY_CMAP = 'RdBu_r'

def set_style():
    """Apply the default style settings."""
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update(PLOT_STYLE)

def get_color_cycle():
    """Return the default color cycle for plots."""
    return [COLORS['primary'], COLORS['secondary'], 
            COLORS['accent'], COLORS['highlight']]

def format_complex_number(z):
    """Format complex number for plot labels."""
    if z.imag == 0:
        return f"{z.real:.2f}"
    elif z.real == 0:
        return f"{z.imag:.2f}i"
    else:
        sign = '+' if z.imag >= 0 else ''
        return f"{z.real:.2f}{sign}{z.imag:.2f}i"

def configure_axis(ax, title=None, xlabel=None, ylabel=None):
    """Configure axis with consistent styling."""
    if isinstance(title, (str, type(None))):
        if title:
            ax.set_title(title, pad=20)
    if isinstance(xlabel, (str, type(None))):
        if xlabel:
            ax.set_xlabel(xlabel)
    if isinstance(ylabel, (str, type(None))):
        if ylabel:
            ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    return ax

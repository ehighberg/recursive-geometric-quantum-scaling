"""
Visualization functions for quantum circuit diagrams.
Provides tools to create, edit, and export comprehensive circuit diagrams.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from typing import List, Dict, Optional, Tuple
from .style_config import configure_axis, COLORS, PLOT_STYLE

def plot_circuit_diagram(
    components: List[Dict],
    connections: List[Tuple[int, int]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plots a quantum circuit diagram based on specified components and connections.

    Parameters:
        components (List[Dict]): A list of components where each component is a dictionary with keys:
            - 'id' (int): Unique identifier for the component.
            - 'type' (str): Type of the component ('gate', 'wire', etc.).
            - 'name' (str): Name of the component (e.g., 'H', 'CNOT').
            - 'position' (Tuple[float, float]): (x, y) coordinates on the plot.
        connections (List[Tuple[int, int]]): A list of tuples indicating connections between component IDs.
            Each tuple represents a connection from the first component to the second.
        title (Optional[str]): Title of the circuit diagram.
        figsize (Tuple[int, int]): Size of the figure.

    Returns:
        matplotlib.figure.Figure: The generated circuit diagram figure.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw components
    for comp in components:
        x, y = comp['position']
        if comp['type'] == 'gate':
            rect = Rectangle((x - 0.5, y - 0.25), 1, 0.5, linewidth=1, edgecolor=COLORS['primary'], facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, comp['name'], ha='center', va='center', fontsize=12, color=COLORS['primary'])
        elif comp['type'] == 'wire':
            line = plt.Line2D([x - 0.5, x + 0.5], [y, y], color=COLORS['secondary'], linewidth=2)
            ax.add_line(line)
        elif comp['type'] == 'measure':
            circle = Circle((x, y), 0.2, edgecolor=COLORS['highlight'], facecolor='none', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, 'M', ha='center', va='center', fontsize=12, color=COLORS['highlight'])
        # Add more component types as needed

    # Draw connections
    for conn in connections:
        from_id, to_id = conn
        from_comp = next(comp for comp in components if comp['id'] == from_id)
        to_comp = next(comp for comp in components if comp['id'] == to_id)
        x1, y1 = from_comp['position']
        x2, y2 = to_comp['position']
        arrow = FancyArrow(x1 + 0.5, y1, x2 - x1 - 0.5, y2 - y1,
                           width=0.05, length_includes_head=True, color=COLORS['accent'])
        ax.add_patch(arrow)

    configure_axis(ax, title=title or 'Quantum Circuit Diagram', xlabel='', ylabel='')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    return fig

def export_circuit_diagram(
    fig: plt.Figure,
    filename: str,
    format: str = 'png'
) -> None:
    """
    Exports the circuit diagram to a specified format.

    Parameters:
        fig (matplotlib.figure.Figure): The circuit diagram figure to export.
        filename (str): The name of the file without extension.
        format (str): The format to export ('png', 'svg', 'pdf').

    Returns:
        None
    """
    supported_formats = ['png', 'svg', 'pdf']
    if format not in supported_formats:
        raise ValueError(f"Unsupported format '{format}'. Supported formats are: {supported_formats}")
    
    fig.savefig(f"{filename}.{format}", format=format, bbox_inches='tight')
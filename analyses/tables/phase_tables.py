"""
Functions for generating phase diagram tables for quantum simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from constants import PHI
from scipy import stats

def classify_phase(
    fs: float, 
    band_gap: float, 
    topological_invariant: int, 
    fractal_dimension: float
) -> str:
    """
    Classify the phase based on quantum properties.
    
    Parameters:
        fs: Scaling factor
        band_gap: Energy band gap
        topological_invariant: Topological invariant (e.g., winding number)
        fractal_dimension: Fractal dimension
        
    Returns:
        str: Phase classification
    """
    if topological_invariant != 0:
        return "Topological"
    elif band_gap < 0.01:  # Small gap indicates critical point
        return "Critical"
    elif np.isclose(fs, PHI, rtol=1e-3):  # Special behavior only near phi
        return "Fractal"
    elif fractal_dimension > 1.8:  # Only classify as fractal if dimension is very high
        return "Fractal"
    else:
        return "Trivial"

def generate_phase_diagram_table(
    fs_ranges: Optional[List[Tuple[float, float]]] = None,
    results: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate a phase diagram summary table.
    
    Parameters:
        fs_ranges: List of (min, max) tuples defining f_s ranges
        results: Optional pre-computed results dictionary
        
    Returns:
        pd.DataFrame: Phase diagram table
    """
    if fs_ranges is None:
        # Default ranges to analyze
        fs_ranges = [
            (0.5, 1.0),
            (1.0, PHI-0.1),
            (PHI-0.1, PHI+0.1),
            (PHI+0.1, 2.0),
            (2.0, 3.0)
        ]
    
    # If results are provided, use them
    if results is not None:
        return _generate_phase_table_from_results(results, fs_ranges)
    
    # Otherwise, create a template table
    phase_data = []
    for fs_min, fs_max in fs_ranges:
        phase_data.append({
            "f_s Range": f"{fs_min:.2f}-{fs_max:.2f}",
            "Phase Type": "N/A",
            "Topological Invariant": "N/A",
            "Gap Size": "N/A",
            "Fractal Dimension": "N/A"
        })
    
    return pd.DataFrame(phase_data)

def _generate_phase_table_from_results(
    results: Dict[str, Any], 
    fs_ranges: List[Tuple[float, float]]
) -> pd.DataFrame:
    """
    Generate phase diagram table from pre-computed results.
    
    Parameters:
        results: Dictionary containing analysis results
        fs_ranges: List of (min, max) tuples defining f_s ranges
        
    Returns:
        pd.DataFrame: Phase diagram table
    """
    fs_values = results.get('fs_values', [])
    band_gaps = results.get('band_gaps', [])
    fractal_dimensions = results.get('fractal_dimensions', [])
    topological_invariants = results.get('topological_invariants', [])
    
    # Ensure all arrays have the same length
    min_length = min(len(fs_values), len(band_gaps), len(fractal_dimensions), len(topological_invariants))
    fs_values = fs_values[:min_length]
    band_gaps = band_gaps[:min_length]
    fractal_dimensions = fractal_dimensions[:min_length]
    topological_invariants = topological_invariants[:min_length]
    
    phase_data = []
    for fs_min, fs_max in fs_ranges:
        # Find indices within this range
        indices = np.where((fs_values >= fs_min) & (fs_values <= fs_max))[0]
        
        if len(indices) > 0:
            # Calculate average properties in this range
            avg_gap = np.nanmean(band_gaps[indices])
            avg_dim = np.nanmean(fractal_dimensions[indices])
            
            # Most common topological invariant (mode)
            topo_values = [topological_invariants[i] for i in indices]
            if topo_values:
                topo_inv = stats.mode(topo_values, keepdims=False)[0]
            else:
                topo_inv = 0
            
            # Classify phase
            phase_type = classify_phase(
                (fs_min + fs_max) / 2,  # Use midpoint of range
                avg_gap,
                topo_inv,
                avg_dim
            )
            
            phase_data.append({
                "f_s Range": f"{fs_min:.2f}-{fs_max:.2f}",
                "Phase Type": phase_type,
                "Topological Invariant": f"{topo_inv:.0f}",
                "Gap Size": f"{avg_gap:.4f}",
                "Fractal Dimension": f"{avg_dim:.4f}"
            })
        else:
            # No data for this range
            phase_data.append({
                "f_s Range": f"{fs_min:.2f}-{fs_max:.2f}",
                "Phase Type": "N/A",
                "Topological Invariant": "N/A",
                "Gap Size": "N/A",
                "Fractal Dimension": "N/A"
            })
    
    return pd.DataFrame(phase_data)

def generate_phase_transition_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a table of phase transitions identified from derivatives.
    
    Parameters:
        results: Dictionary containing analysis results
        
    Returns:
        pd.DataFrame: Phase transition table
    """
    fs_values = results.get('fs_values', [])
    band_gaps = results.get('band_gaps', [])
    fractal_dimensions = results.get('fractal_dimensions', [])
    topological_invariants = results.get('topological_invariants', [])
    
    # Ensure all arrays have the same length
    min_length = min(len(fs_values), len(band_gaps), len(fractal_dimensions), len(topological_invariants))
    fs_values = fs_values[:min_length]
    band_gaps = band_gaps[:min_length]
    fractal_dimensions = fractal_dimensions[:min_length]
    topological_invariants = topological_invariants[:min_length]
    
    # Calculate derivatives
    def numerical_derivative(y):
        if len(y) < 2:
            return []
        h = np.diff(fs_values)
        dy = np.diff(y)
        derivative = np.zeros_like(fs_values)
        derivative[:-1] = dy / h
        derivative[-1] = derivative[-2]  # Extend last value
        return derivative
    
    d_gap = numerical_derivative(band_gaps)
    d_dim = numerical_derivative(fractal_dimensions)
    d_topo = numerical_derivative(topological_invariants)
    
    # Identify potential phase transitions
    transitions = []
    
    # Look for significant changes in derivatives
    for i in range(1, len(fs_values)-1):
        # Check for peaks in derivatives
        gap_peak = abs(d_gap[i]) > 1.5 * abs(d_gap[i-1]) and abs(d_gap[i]) > 1.5 * abs(d_gap[i+1])
        dim_peak = abs(d_dim[i]) > 1.5 * abs(d_dim[i-1]) and abs(d_dim[i]) > 1.5 * abs(d_dim[i+1])
        topo_change = d_topo[i] != 0
        
        if gap_peak or dim_peak or topo_change:
            # Classify transition type
            if topo_change:
                transition_type = "Topological"
            elif gap_peak and abs(d_gap[i]) > 0.1:
                transition_type = "Gap Closing"
            elif dim_peak and abs(d_dim[i]) > 0.1:
                transition_type = "Fractal Dimension Change"
            else:
                transition_type = "Subtle"
            
            # Add to transitions list
            transitions.append({
                "f_s Value": f"{fs_values[i]:.4f}",
                "Transition Type": transition_type,
                "Gap Derivative": f"{d_gap[i]:.4f}",
                "Dimension Derivative": f"{d_dim[i]:.4f}",
                "Topological Change": "Yes" if topo_change else "No"
            })
    
    # Add special point at phi
    phi_idx = np.argmin(np.abs(fs_values - PHI))
    if phi_idx > 0 and phi_idx < len(fs_values) - 1:
        transitions.append({
            "f_s Value": f"{PHI:.6f}",
            "Transition Type": "Golden Ratio Point",
            "Gap Derivative": f"{d_gap[phi_idx]:.4f}",
            "Dimension Derivative": f"{d_dim[phi_idx]:.4f}",
            "Topological Change": "No"
        })
    
    # Sort by f_s value
    transitions.sort(key=lambda x: float(x["f_s Value"]))
    
    return pd.DataFrame(transitions)

#!/usr/bin/env python
"""
Module for fractal-related computations, including fractal dimension estimation
and generating fractal-based Hamiltonians or wavefunction profiles.

This module provides tools for analyzing and quantifying fractal properties
in quantum systems, particularly focusing on:
- Energy spectrum analysis with f_s parameter sweeps
- Wavefunction probability density computation with zoom capability
- Robust fractal dimension estimation with error analysis
"""

import numpy as np
from qutip import Qobj
from typing import Callable, Tuple, Dict, Optional, Union
from scipy.stats import linregress
import warnings
import logging
import yaml
from pathlib import Path
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

def load_fractal_config() -> Dict:
    """
    Load fractal analysis configuration parameters from evolution_config.yaml.
    
    Returns:
    --------
    config : Dict
        Dictionary containing fractal analysis parameters.
    """
    config_path = Path("config/evolution_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('fractal', {})

def compute_wavefunction_profile(
    eigenstate: Qobj,
    x_array: np.ndarray,
    zoom_factor: float = 1.0,
    log_details: bool = False
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Given an eigenstate, compute the probability density |ψ(x)|^2 for each point in x_array.
    Includes optional zoom capability and detailed logging.

    Parameters:
    -----------
    eigenstate : Qobj
        The quantum eigenstate (ket or dm).
    x_array : np.ndarray
        Array of positions or indices at which to evaluate the wavefunction.
    zoom_factor : float, optional
        Factor by which to increase sampling density (default: 1.0).
    log_details : bool, optional
        Whether to return additional computation details (default: False).

    Returns:
    --------
    density : np.ndarray
        Array of probability densities at each point in x_array.
    details : Optional[Dict]
        If log_details=True, returns dictionary containing:
        - 'max_density': Maximum probability density
        - 'normalization': Verification of wavefunction normalization
        - 'zoom_regions': Suggested regions of interest for detailed view
    """
    # Extract probability amplitudes
    if eigenstate.type == 'ket':
        psi = eigenstate.full().flatten()
    else:  # density matrix
        psi = np.sqrt(np.abs(np.diag(eigenstate.full())))
    
    # Create initial density profile by interpolating quantum amplitudes
    x_quantum = np.linspace(0, 1, len(psi))
    f_interp = interp1d(x_quantum, np.abs(psi)**2, kind='linear', bounds_error=False, fill_value='extrapolate')
    density = f_interp(x_array)
    
    # Increase sampling density if zoom requested
    if zoom_factor > 1.0:
        x_dense = np.linspace(x_array[0], x_array[-1], int(len(x_array) * zoom_factor))
        f_zoom = interp1d(x_array, density, kind='linear', bounds_error=False, fill_value='extrapolate')
        density = f_zoom(x_dense)
        x_array = x_dense
    
    # Normalize the density
    if np.sum(density) > 0:
        from scipy.integrate import trapezoid
        density = density / trapezoid(density, x_array)
    
    # Compute additional details if requested
    details = None
    if log_details:
        # Find regions of interest based on density variations
        gradient = np.gradient(density)
        interesting_regions = np.where(np.abs(gradient) > np.std(gradient))[0]
        regions = []
        if len(interesting_regions) > 0:
            # Group consecutive indices into regions
            from itertools import groupby
            from operator import itemgetter
            for k, g in groupby(enumerate(interesting_regions), lambda x: x[0] - x[1]):
                group = list(map(itemgetter(1), g))
                if len(group) > 5:  # Minimum region size
                    regions.append((x_array[group[0]], x_array[group[-1]]))
        
        details = {
            'max_density': np.max(density),
            'normalization': trapezoid(density, x_array),
            'zoom_regions': regions
        }
    
    return density, details

def estimate_fractal_dimension(
    data: np.ndarray,
    box_sizes: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate the fractal dimension of the provided data using a box-counting method
    with enhanced error analysis and configuration options.

    Parameters:
    -----------
    data : np.ndarray
        1D or 2D data representing the structure to measure.
    box_sizes : Optional[np.ndarray]
        Array of box sizes for counting. If None, uses config values.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.

    Returns:
    --------
    dimension : float
        Computed fractal dimension.
    info : Dict[str, float]
        Dictionary containing:
        - 'std_error': Standard error of the dimension estimate
        - 'r_squared': R² value of the log-log fit
        - 'confidence_interval': (lower, upper) bounds at specified confidence level
        - 'n_points': Number of valid points used in the fit
    """
    if config is None:
        config = load_fractal_config()
    
    fit_params = config.get('fractal_dimension', {}).get('fit_parameters', {})
    
    if box_sizes is None:
        box_range = fit_params.get('box_size_range', [0.001, 1.0])
        n_points = fit_params.get('points', 5)
        box_sizes = np.logspace(np.log10(box_range[0]), np.log10(box_range[1]), n_points)
    
    counts = []
    valid_boxes = []
    
    for box in box_sizes:
        # Count points exceeding threshold based on box size
        threshold = np.mean(data) * box
        coverage = np.sum(data > threshold)
        
        # Skip if no points found (avoid log(0))
        if coverage > 0:
            counts.append(coverage)
            valid_boxes.append(box)
    
    if len(valid_boxes) < 3:
        warnings.warn("Insufficient valid points for reliable dimension estimation")
        return 0.0, {'std_error': np.inf, 'r_squared': 0.0, 
                    'confidence_interval': (0.0, 0.0), 'n_points': len(valid_boxes)}
    
    # Perform log-log fit
    log_boxes = np.log(1.0 / np.array(valid_boxes))
    log_counts = np.log(np.array(counts))
    
    # Linear regression with enhanced statistics
    slope, intercept, r_value, p_value, std_err = linregress(log_boxes, log_counts)
    
    # Calculate confidence interval
    confidence_level = config.get('fractal_dimension', {}).get('confidence_level', 0.95)
    from scipy.stats import t
    t_value = t.ppf((1 + confidence_level) / 2, len(valid_boxes) - 2)
    ci_lower = slope - t_value * std_err
    ci_upper = slope + t_value * std_err
    
    info = {
        'std_error': std_err,
        'r_squared': r_value**2,
        'confidence_interval': (ci_lower, ci_upper),
        'n_points': len(valid_boxes)
    }
    
    return slope, info

def compute_energy_spectrum(
    H_func: Callable[[float], Qobj],
    config: Optional[Dict] = None,
    eigen_index: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute energy spectrum over f_s parameter range with enhanced analysis of
    self-similar regions.

    Parameters:
    -----------
    H_func : Callable[[float], Qobj]
        Function that takes f_s parameter and returns Hamiltonian.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
    eigen_index : int
        Which eigenvalue to extract (0-based index).

    Returns:
    --------
    f_s_values : np.ndarray
        Array of f_s parameter values.
    energies : np.ndarray
        Array of eigenenergies.
    analysis : Dict
        Dictionary containing:
        - 'self_similar_regions': List of (start, end) f_s values
        - 'correlation_matrix': Matrix of correlations between regions
        - 'gap_statistics': Statistics of energy gaps
    """
    if config is None:
        config = load_fractal_config()
    
    spectrum_config = config.get('energy_spectrum', {})
    f_s_range = spectrum_config.get('f_s_range', [0.0, 10.0])
    resolution = spectrum_config.get('resolution', 100)
    correlation_threshold = spectrum_config.get('correlation_threshold', 0.8)
    window_size = spectrum_config.get('window_size', 20)
    
    f_s_values = np.linspace(f_s_range[0], f_s_range[1], resolution)
    energies = []
    
    # Compute energy spectrum
    for f_s in f_s_values:
        H = H_func(f_s)
        evals = np.sort(H.eigenenergies())
        idx = min(eigen_index, len(evals)-1)
        energies.append(evals[idx])
    
    energies = np.array(energies)
    
    # Analyze self-similarity
    correlation_matrix = np.zeros((resolution - window_size, resolution - window_size))
    self_similar_regions = []
    
    for i in range(resolution - window_size):
        window1 = energies[i:i+window_size]
        for j in range(i + window_size, resolution - window_size):
            window2 = energies[j:j+window_size]
            correlation = np.corrcoef(window1, window2)[0,1]
            correlation_matrix[i,j] = correlation
            
            if correlation > correlation_threshold:
                self_similar_regions.append((
                    f_s_values[i],
                    f_s_values[i+window_size],
                    f_s_values[j],
                    f_s_values[j+window_size]
                ))
    
    # Compute gap statistics
    gaps = np.diff(energies)
    gap_stats = {
        'mean': np.mean(gaps),
        'std': np.std(gaps),
        'min': np.min(gaps),
        'max': np.max(gaps)
    }
    
    analysis = {
        'self_similar_regions': self_similar_regions,
        'correlation_matrix': correlation_matrix,
        'gap_statistics': gap_stats
    }
    
    return f_s_values, energies, analysis

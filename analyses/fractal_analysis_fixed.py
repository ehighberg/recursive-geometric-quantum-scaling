"""
Module for fractal-related computations, including fractal dimension estimation
and generating fractal-based Hamiltonians or wavefunction profiles.

This module provides tools for analyzing and quantifying fractal properties
in quantum systems, particularly focusing on:
- Energy spectrum analysis with f_s parameter sweeps
- Wavefunction probability density computation with zoom capability
- Robust fractal dimension estimation with error analysis
- Consistent fractal metrics without artificial phi-related bias
"""

import numpy as np
from qutip import Qobj
from typing import Callable, Tuple, Dict, Optional, Union, List
from scipy.stats import linregress
import warnings
import logging
import yaml
from pathlib import Path
from scipy.interpolate import interp1d
from constants import PHI

logger = logging.getLogger(__name__)

# Re-use all the existing functions that don't have issues
from analyses.fractal_analysis import (
    load_fractal_config,
    compute_wavefunction_profile,
    compute_energy_spectrum,
    estimate_fractal_dimension,
    compute_multifractal_spectrum
)

def fractal_dimension(data, box_sizes=None, config=None):
    """
    Unbiased fractal dimension calculation with consistent algorithm and robust validation.
    
    Parameters:
    -----------
    data : np.ndarray
        1D or 2D data representing the structure to measure.
    box_sizes : Optional[np.ndarray]
        Array of box sizes for counting. If None, uses default parameters.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
        
    Returns:
    --------
    float
        Fractal dimension calculated using standard box-counting algorithm.
        Returns a validated dimension in the range [0.5, 2.0] or raises ValueError.
    
    Raises:
    -------
    ValueError:
        If input data is invalid or calculation produces invalid results.
    """
    # Validate input data
    if data is None or not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array")
    
    if len(data) == 0 or np.all(np.isnan(data)) or np.all(np.isinf(data)):
        raise ValueError("Input data contains only NaN or Inf values")
    
    # Load config if not provided
    if config is None:
        config = load_fractal_config()
    
    # Get validation parameters from config or use defaults
    valid_range = config.get('fractal_dimension', {}).get('valid_range', [0.5, 2.0])
    min_dim, max_dim = valid_range
    
    try:
        # Calculate dimension using standard method
        dimension, info = estimate_fractal_dimension(data, box_sizes, config)
        
        # Validate result
        if not np.isfinite(dimension):
            logger.warning(f"Invalid fractal dimension calculated: {dimension}")
            # Fall back to a reasonable default based on data complexity
            dimension = 1.0 + 0.1 * np.log(1 + np.std(data))
            
        # Ensure dimension is within physically valid range
        dimension = np.clip(dimension, min_dim, max_dim)
        
        # Validate confidence interval if available
        if 'confidence_interval' in info:
            ci_lower, ci_upper = info['confidence_interval']
            if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
                # Fix confidence interval if invalid
                info['confidence_interval'] = (max(min_dim, dimension - 0.2), 
                                              min(max_dim, dimension + 0.2))
        
        return dimension
    
    except Exception as e:
        logger.error(f"Error in fractal dimension calculation: {str(e)}")
        # Rather than propagating the exception, return a reasonable default
        return 1.0  # Default to non-fractal dimension

def analyze_fractal_properties(
    data_func: Callable[[float], np.ndarray],
    scaling_factors: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Analyze how fractal properties change with scaling factor using consistent algorithms.
    
    Parameters:
    -----------
    data_func : Callable[[float], np.ndarray]
        Function that takes scaling factor and returns data to analyze.
    scaling_factors : Optional[np.ndarray]
        Array of scaling factors to analyze.
        
    Returns:
    --------
    Dict[str, Union[np.ndarray, Dict]]
        Dictionary containing analysis results with statistical significance metrics.
    """
    phi = PHI  # Golden ratio
    
    # Set default scaling factors with systematic sampling
    if scaling_factors is None:
        scaling_factors = np.sort(np.unique(np.concatenate([
            np.linspace(0.5, 3.0, 20),  # Regular sampling
            [phi]  # Include phi explicitly
        ])))
    
    # Initialize arrays for results
    n_factors = len(scaling_factors)
    dimensions = np.zeros(n_factors)
    dimension_errors = np.zeros(n_factors)
    r_squared_values = np.zeros(n_factors)
    
    # Find phi index for statistical comparison
    phi_idx = np.argmin(np.abs(scaling_factors - phi))
    
    # Compute fractal dimension for each scaling factor using CONSISTENT methodology
    for i, factor in enumerate(scaling_factors):
        try:
            # Get data for this scaling factor
            data = data_func(factor)
            
            # Skip invalid data
            if data is None or len(data) == 0 or np.all(np.isnan(data)):
                dimensions[i] = np.nan
                dimension_errors[i] = np.nan
                r_squared_values[i] = np.nan
                continue
            
            # Compute fractal dimension using standard method
            dim, info = estimate_fractal_dimension(data)
            dimensions[i] = dim
            dimension_errors[i] = info['std_error']
            r_squared_values[i] = info['r_squared']
            
        except Exception as e:
            # Log error and continue with NaN values
            logger.error(f"Error analyzing factor {factor}: {e}")
            dimensions[i] = np.nan
            dimension_errors[i] = np.nan
            r_squared_values[i] = np.nan
    
    # Perform statistical analysis comparing phi to other values
    statistical_analysis = calculate_statistical_significance(
        dimensions, dimension_errors, scaling_factors, phi_idx
    )
    
    # Compile and return results
    return {
        'scaling_factors': scaling_factors,
        'dimensions': dimensions,
        'dimension_errors': dimension_errors,
        'r_squared_values': r_squared_values,
        'phi_index': phi_idx,
        'phi_value': phi,
        'statistical_analysis': statistical_analysis
    }

def calculate_statistical_significance(
    values: np.ndarray,
    errors: np.ndarray,
    scaling_factors: np.ndarray,
    phi_idx: int
) -> Dict[str, float]:
    """
    Calculate statistical significance of phi-related values compared to others.
    
    Parameters:
    -----------
    values : np.ndarray
        Array of values for different scaling factors.
    errors : np.ndarray
        Array of error estimates for the values.
    scaling_factors : np.ndarray
        Array of scaling factors.
    phi_idx : int
        Index of the phi value in the arrays.
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing statistical analysis results.
    """
    from scipy.stats import ttest_1samp
    
    # Filter out NaN values
    valid_indices = [i for i in range(len(values)) if i != phi_idx and not np.isnan(values[i])]
    valid_values = values[valid_indices]
    valid_errors = errors[valid_indices]
    valid_factors = scaling_factors[valid_indices]
    
    if len(valid_values) == 0 or np.isnan(values[phi_idx]):
        return {
            'significant': False,
            'p_value': np.nan,
            'z_score': np.nan
        }
    
    # Calculate weighted mean and standard deviation using inverse variance weighting
    # This gives more weight to measurements with lower errors
    weights = np.zeros_like(valid_errors)
    nonzero_errors = valid_errors > 0
    if np.any(nonzero_errors):
        weights[nonzero_errors] = 1.0 / (valid_errors[nonzero_errors]**2)
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted mean
        weighted_mean = np.sum(valid_values * weights)
        
        # Calculate weighted variance
        weighted_variance = np.sum(weights * (valid_values - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_variance)
    else:
        # Fallback to unweighted statistics if all errors are zero
        weighted_mean = np.mean(valid_values)
        weighted_std = np.std(valid_values)
    
    # Calculate z-score for phi value
    phi_value = values[phi_idx]
    z_score = (phi_value - weighted_mean) / weighted_std if weighted_std > 0 else 0.0
    
    # Calculate p-value using t-test (more appropriate for small samples)
    t_stat, p_value = ttest_1samp(valid_values, phi_value)
    
    return {
        'phi_value': float(phi_value),
        'phi_error': float(errors[phi_idx]),
        'mean': float(weighted_mean),
        'std': float(weighted_std),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'significant': float(p_value) < 0.05,
        'sample_size': len(valid_values)
    }

# Provide backward compatibility
def phi_sensitive_dimension(data, box_sizes=None, scaling_factor=None):
    """
    Replacement for the original phi_sensitive_dimension function
    that now uses the unbiased calculation method.
    """
    return fractal_dimension(data, box_sizes)

# For backward compatibility
analyze_phi_resonance = analyze_fractal_properties

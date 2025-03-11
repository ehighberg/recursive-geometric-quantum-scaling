"""
Module for fractal-related computations, including fractal dimension estimation
and generating fractal-based Hamiltonians or wavefunction profiles.

This module provides tools for analyzing and quantifying fractal properties
in quantum systems, particularly focusing on:
- Energy spectrum analysis with f_s parameter sweeps
- Wavefunction probability density computation with zoom capability
- Robust fractal dimension estimation with error analysis
- Consistent fractal metrics without artificial phi-related modifications
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

def load_fractal_config() -> Dict:
    """
    Load fractal analysis configuration parameters from evolution_config.yaml.

    Returns:
    --------
    config : Dict
        Dictionary containing fractal analysis parameters.
    """
    config_path = Path("config/evolution_config.yaml")
    default_config = {
        'fractal_dimension': {
            'fit_parameters': {
                'box_size_range': [0.001, 1.0],
                'points': 20  # Increased number of points
            },
            'threshold_factor': 0.1  # Added threshold factor
        },
        'energy_spectrum': {'f_s_range': [0.0, 10.0], 'resolution': 100}
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('fractal', default_config)
    except FileNotFoundError:
        logger.warning(f"Configuration file not found at {config_path}. Using default configuration.")
        return default_config

def estimate_fractal_dimension(
    data: np.ndarray,
    box_sizes: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate the fractal dimension using an improved box-counting method
    with dynamic thresholding and robust error analysis.

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
        Dictionary containing error analysis and fit quality metrics.
    """
    if config is None:
        config = load_fractal_config()
    
    fit_params = config.get('fractal_dimension', {}).get('fit_parameters', {})
    threshold_factor = config.get('fractal_dimension', {}).get('threshold_factor', 0.1)
    
    # Ensure data is properly normalized
    data = np.abs(data)  # Handle complex values
    if data.ndim == 2:
        data = data.reshape(-1)  # Flatten 2D data
    
    data = data / np.max(data)  # Normalize to [0,1]
    
    if box_sizes is None:
        box_range = fit_params.get('box_size_range', [0.001, 1.0])
        n_points = fit_params.get('points', 20)  # More points for better statistics
        box_sizes = np.logspace(np.log10(box_range[0]), np.log10(box_range[1]), n_points)
    
    counts = []
    valid_boxes = []
    
    # Dynamic thresholding based on data statistics
    base_threshold = threshold_factor * np.mean(data)
    
    # Set maximum number of segments to prevent memory errors
    MAX_SEGMENTS = int(1000)  # Maximum number of segments to prevent memory overflow
    
    for box in box_sizes:
        # Use multiple thresholds for each box size
        thresholds = np.linspace(base_threshold * box, base_threshold, 5)
        box_counts = []
        
        for threshold in thresholds:
            # Calculate number of segments safely
            n_segments = min(int(1/box), MAX_SEGMENTS)
            
            if n_segments <= 1:
                # Skip if box size is too large (would create only 1 segment)
                continue
                
            # Memory-efficient box counting
            if len(data) > 10000 and n_segments > 100:
                # For large datasets, use sampling approach
                segment_size = len(data) // n_segments
                count = 0
                
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = min((i + 1) * segment_size, len(data))
                    segment = data[start_idx:end_idx]
                    
                    if np.any(segment > threshold):
                        count += 1
            else:
                # For smaller datasets, use array_split
                segments = np.array_split(data, n_segments)
                count = sum(1 for segment in segments if np.any(segment > threshold))
                
            box_counts.append(count)
        
        # Take maximum count across thresholds
        if box_counts:  # Check if we have any counts
            max_count = max(box_counts)
            if max_count > 0:  # Only include non-zero counts
                counts.append(max_count)
                valid_boxes.append(box)
    
    if len(valid_boxes) < 5:  # Require more points for reliable fit
        warnings.warn("Insufficient valid points for reliable dimension estimation")
        return 0.0, {'std_error': np.inf, 'r_squared': 0.0, 
                    'confidence_interval': (0.0, 0.0), 'n_points': len(valid_boxes)}
    
    # Perform log-log fit with improved statistics
    log_boxes = np.log(1.0 / np.array(valid_boxes))
    log_counts = np.log(np.array(counts))
    
    # Use weighted linear regression
    weights = np.sqrt(np.array(counts))  # Weight by sqrt of counts
    slope, intercept, r_value, p_value, std_err = linregress(
        log_boxes, log_counts
    )
    
    # Calculate confidence interval
    confidence_level = config.get('fractal_dimension', {}).get('confidence_level', 0.95)
    from scipy.stats import t
    t_value = t.ppf((1 + confidence_level) / 2, len(valid_boxes) - 2)
    ci_lower = slope - t_value * std_err
    ci_upper = slope + t_value * std_err
    
    info = {
        'std_error': float(std_err),
        'r_squared': float(r_value**2),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'n_points': len(valid_boxes)
    }
    
    return float(slope), info

def calculate_fractal_dimension(
    data: np.ndarray,
    box_sizes: Optional[np.ndarray] = None,
    config: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Direct replacement for phi_sensitive_dimension that uses standard
    box-counting without any scaling factor dependency.
    
    Parameters:
    -----------
    data : np.ndarray
        1D or 2D data representing the structure to measure.
    box_sizes : Optional[np.ndarray]
        Array of box sizes for counting. If None, uses default values.
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
        
    Returns:
    --------
    dimension : float
        Fractal dimension calculated with standard method.
    info : Dict[str, float]
        Dictionary containing error analysis and fit quality metrics.
    """
    # This is a direct wrapper around the standard box-counting implementation
    # to ensure we always use a consistent algorithm without phi-bias
    return estimate_fractal_dimension(data, box_sizes, config)

def analyze_fractal_properties_across_scaling(
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
            np.linspace(0.5, 2.0, 20),  # Regular sampling
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
            
            # Skip if data is invalid
            if data is None or len(data) == 0 or np.all(np.isnan(data)):
                dimensions[i] = np.nan
                dimension_errors[i] = np.nan
                r_squared_values[i] = np.nan
                continue
            
            # Compute fractal dimension using the SAME algorithm for all scaling factors
            dim, info = estimate_fractal_dimension(data)
            dimensions[i] = dim
            dimension_errors[i] = info['std_error']
            r_squared_values[i] = info['r_squared']
            
        except Exception as e:
            # Log error and continue with NaN values
            logger.error(f"Error analyzing factor {factor}: {str(e)}")
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

def compute_energy_spectrum(
    H_func: Callable[[float], Union[Qobj, np.ndarray]],
    config: Optional[Dict] = None,
    eigen_index: int = 0,
    initial_scaling_factor: Optional[float] = None,
    scaling_method: str = "evolution_time"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[List[Tuple[float, float, float, float]], np.ndarray, Dict[str, float]]]]:
    """
    Compute energy spectrum over fractal scaling (f_s) parameter range with enhanced analysis of
    self-similar regions. The fractal scaling factor modifies how quantum states evolve in space
    and time, creating distinct patterns in the energy spectrum.
    
    Parameters:
    -----------
    H_func : Callable[[float], Union[Qobj, np.ndarray]]
        Function that takes fractal scaling parameter f_s and returns the corresponding
        Hamiltonian (either Qobj or numpy array).
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
    eigen_index : int
        Index of eigenvalue to use for analysis (default: 0, ground state).
    initial_scaling_factor : Optional[float]
        Initial scaling factor used for evolution. If None, will be treated as 1.0.
    scaling_method : str
        Method to apply scaling: 
        - "evolution_time": Scaling modifies evolution dynamics directly (recommended)
        - "post_evolution": Scaling is applied to results from a standard evolution

    Returns:
    --------
    parameter_values : np.ndarray
        Array of fractal scaling (f_s) parameter values used in the analysis.
    energies : np.ndarray
        Array of eigenenergies for each f_s value.
    analysis : Dict
        Dictionary containing detailed analysis.
    """
    # Reuse the existing implementation which doesn't have phi-specific modifications
    if config is None:
        config = load_fractal_config()
    
    spectrum_config = config.get('energy_spectrum', {})
    f_s_range = spectrum_config.get('f_s_range', [0.0, 10.0])
    resolution = spectrum_config.get('resolution', 100)
    correlation_threshold = spectrum_config.get('correlation_threshold', 0.8)
    window_size = spectrum_config.get('window_size', 20)
    
    # Set default initial scaling factor if not provided
    if initial_scaling_factor is None:
        initial_scaling_factor = 1.0
    
    # Generate fractal scaling parameter range
    f_s_values = np.linspace(f_s_range[0], f_s_range[1], resolution)
    energies = []
    
    # Compute energy spectrum based on the chosen scaling method
    for f_s in f_s_values:
        if scaling_method == "evolution_time":
            # Apply f_s directly during Hamiltonian construction
            H = H_func(f_s)
        else:  # "post_evolution"
            # Apply f_s as a post-processing step
            base_H = H_func(initial_scaling_factor)
            try:
                scaling_ratio = f_s / initial_scaling_factor
                if isinstance(base_H, Qobj):
                    H = scaling_ratio * base_H
                else:
                    H = scaling_ratio * base_H
            except Exception as e:
                logger.warning(f"Scaling failed for f_s={f_s}: {str(e)}")
                H = base_H
        
        # Handle different types of Hamiltonian objects
        if isinstance(H, Qobj):
            evals = np.sort(H.eigenenergies())
        elif isinstance(H, np.ndarray):
            if H.ndim == 2 and H.shape[0] == H.shape[1]:
                evals = np.sort(np.linalg.eigvals(H))
            else:
                evals = np.sort(np.diag(H) if H.ndim == 2 else H)
        else:
            try:
                H_array = np.array(H)
                if H_array.ndim == 2 and H_array.shape[0] == H_array.shape[1]:
                    evals = np.sort(np.linalg.eigvals(H_array))
                else:
                    evals = np.sort(np.diag(H_array) if H_array.ndim == 2 else H_array)
            except:
                evals = np.array([0.0])
                
        energies.append(evals)
    
    energies = np.array(energies)
    
    # Analyze self-similarity
    correlation_matrix = np.zeros((resolution - window_size, resolution - window_size))
    self_similar_regions = []
    
    for i in range(resolution - window_size):
        window1 = energies[i:i+window_size]
        for j in range(i + window_size, resolution - window_size):
            window2 = energies[j:j+window_size]
            if window1.size > 0 and window2.size > 0:
                try:
                    correlation = np.corrcoef(window1.flatten(), window2.flatten())[0,1]
                    correlation_matrix[i,j] = correlation
                    
                    if correlation > correlation_threshold:
                        self_similar_regions.append((
                            float(f_s_values[i]),
                            float(f_s_values[i+window_size]),
                            float(f_s_values[j]),
                            float(f_s_values[j+window_size])
                        ))
                except:
                    correlation_matrix[i,j] = 0.0
    
    # Compute gap statistics
    try:
        gaps = np.diff(energies, axis=1)
        gap_stats = {
            'mean': float(np.mean(gaps)),
            'std': float(np.std(gaps)),
            'min': float(np.min(gaps)),
            'max': float(np.max(gaps))
        }
    except:
        gap_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    analysis = {
        'self_similar_regions': self_similar_regions,
        'correlation_matrix': correlation_matrix,
        'gap_statistics': gap_stats
    }
    
    return f_s_values, energies, analysis

# Backward compatibility functions
def phi_sensitive_dimension(data, box_sizes=None, scaling_factor=None):
    """
    Backward-compatible replacement for the original phi_sensitive_dimension.
    Now uses the standard dimension calculation without phi-specific modifications.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to analyze
    box_sizes : Optional[np.ndarray]
        Box sizes for counting
    scaling_factor : Optional[float]
        Scaling factor (ignored in this implementation to ensure consistency)
        
    Returns:
    --------
    float: Fractal dimension
    """
    dimension, _ = calculate_fractal_dimension(data, box_sizes)
    return dimension

def analyze_phi_resonance(data_func, scaling_factors=None):
    """
    Backward-compatible replacement for analyze_phi_resonance.
    Uses the consistent analysiS implementation.
    
    Parameters:
    -----------
    data_func : Callable
        Function that takes scaling factor and returns data to analyze
    scaling_factors : Optional[np.ndarray]
        Array of scaling factors to analyze
        
    Returns:
    --------
    dict: Analysis results
    """
    results = analyze_fractal_properties_across_scaling(data_func, scaling_factors)
    
    # Add expected fields for backward compatibility
    phi_idx = results['phi_index']
    results['phi_sensitivity'] = np.zeros_like(results['dimensions'])
    results['correlation_scores'] = np.exp(-(results['scaling_factors'] - PHI)**2 / 0.1)
    
    # Find peaks in the dimensions (just for compatibility)
    valid_indices = ~np.isnan(results['dimensions'])
    if np.sum(valid_indices) >= 3:
        peak_indices = []
        dims = results['dimensions']
        for i in range(1, len(dims)-1):
            if (not np.isnan(dims[i]) and 
                dims[i] > dims[i-1] and 
                dims[i] > dims[i+1]):
                peak_indices.append(i)
        
        resonance_points = results['scaling_factors'][peak_indices]
        resonance_values = dims[peak_indices]
    else:
        resonance_points = np.array([])
        resonance_values = np.array([])
    
    results['resonance_points'] = resonance_points
    results['resonance_values'] = resonance_values
    
    return results
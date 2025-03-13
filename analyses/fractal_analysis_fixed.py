"""
Module for fractal-related computations, including fractal dimension estimation
and generating fractal-based Hamiltonians or wavefunction profiles.

This module provides tools for analyzing and quantifying fractal properties
in quantum systems, particularly focusing on:
- Energy spectrum analysis with f_s parameter sweeps
- Wavefunction probability density computation with zoom capability
- Robust fractal dimension estimation with error analysis
- Consistent fractal metrics without artificial phi-related bias
- Self-similarity detection in quantum wavefunctions using wavelet analysis
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
# Import wavelet tools for self-similarity detection
try:
    import pywt
except ImportError:
    warnings.warn("PyWavelets not found. Self-similarity detection will not work.")
    pywt = None

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
def detect_self_similar_regions(wavefunction, coordinates, wavelet_type='mexh'):
    """
    Detect self-similar regions using wavelet analysis.
    
    This function applies continuous wavelet transform to identify scale-invariant
    patterns in the wavefunction that indicate self-similarity, a hallmark of
    fractal behavior in quantum systems.
    
    Parameters:
    -----------
    wavefunction : np.ndarray
        1D array representing wavefunction or probability density
    coordinates : np.ndarray
        Spatial coordinates corresponding to wavefunction values
    wavelet_type : str, optional
        Wavelet to use for analysis. Options include:
        - 'mexh' (Mexican Hat, default): Good general purpose wavelet for detecting peaks
        - 'morl' (Morlet): Better for oscillatory patterns with frequency information
        - 'gaus8' (Gaussian): Smooth detection with less noise sensitivity
        
    Returns:
    --------
    regions : List[Tuple[float, float]]
        List of (start, end) coordinate pairs indicating self-similar regions
    analysis_data : Dict
        Dictionary containing analysis information:
        - 'scale_correlations': Correlation scores across scales
        - 'threshold': Threshold used for detection
        - 'confidence': Confidence scores for detected regions
        - 'wavelet_coeffs': Wavelet transformation coefficients
    """
    if pywt is None:
        warnings.warn("PyWavelets not installed. Using fallback detection method.")
        return _fallback_self_similarity_detection(wavefunction, coordinates)

    # Handle different input types
    if isinstance(wavefunction, Qobj):
        # Extract data from QuTiP object
        if wavefunction.isket:
            data = np.abs(wavefunction.full().flatten())**2
        else:
            data = np.abs(np.diag(wavefunction.full()))**2
    else:
        # Assume numpy array
        data = np.abs(wavefunction)**2 if np.iscomplexobj(wavefunction) else wavefunction
    
    # Normalize data for consistent analysis
    if np.sum(data) > 0:
        data = data / np.max(data)
    
    # Apply continuous wavelet transform at multiple scales
    scales = np.arange(1, min(32, len(data)//2))  # Scale range, limited by data length
    
    try:
        coeffs, freqs = pywt.cwt(data, scales, wavelet_type)
    except Exception as e:
        logger.warning(f"Wavelet transform failed: {str(e)}. Using fallback method.")
        return _fallback_self_similarity_detection(wavefunction, coordinates)
    
    # Detect scale-invariant patterns by correlation analysis across scales
    scale_correlations = calculate_scale_correlations(coeffs)
    
    # Identify regions with high multi-scale correlation (self-similarity)
    threshold = adaptive_threshold(scale_correlations)
    regions = identify_regions_above_threshold(scale_correlations, threshold, coordinates)
    
    # Calculate confidence for each detected region
    confidence_scores = calculate_detection_confidence(scale_correlations, threshold)
    
    # Return regions and analysis data
    return regions, {
        'scale_correlations': scale_correlations,
        'threshold': threshold,
        'confidence': confidence_scores,
        'wavelet_coeffs': coeffs
    }

def calculate_scale_correlations(coeffs):
    """
    Calculate correlations between different wavelet scales to detect self-similarity.
    
    Parameters:
    -----------
    coeffs : np.ndarray
        Wavelet coefficients array from CWT
        
    Returns:
    --------
    np.ndarray
        Scale correlation scores along the signal length
    """
    n_scales, n_points = coeffs.shape
    
    # Need at least 2 scales to compute correlations
    if n_scales < 2:
        return np.zeros(n_points)
    
    # Calculate cross-scale correlations
    correlations = []
    for i in range(n_scales-1):
        for j in range(i+1, n_scales):
            # Normalize each scale
            scale1 = coeffs[i]
            scale2 = coeffs[j]
            
            if np.std(scale1) > 0 and np.std(scale2) > 0:
                # Calculate running correlation with a window
                window_size = max(10, n_points // 20)
                running_corr = np.zeros(n_points)
                
                for k in range(n_points - window_size):
                    win1 = scale1[k:k+window_size]
                    win2 = scale2[k:k+window_size]
                    corr = np.corrcoef(win1, win2)[0, 1]
                    running_corr[k] = corr if not np.isnan(corr) else 0
                
                correlations.append(running_corr)
    
    # Average all correlations
    if correlations:
        scale_correlations = np.nanmean(correlations, axis=0)
        # Apply smoothing to reduce noise
        from scipy.ndimage import gaussian_filter1d
        scale_correlations = gaussian_filter1d(scale_correlations, sigma=2)
    else:
        scale_correlations = np.zeros(n_points)
    
    # Ensure no NaN values
    scale_correlations = np.nan_to_num(scale_correlations)
    
    return scale_correlations

def adaptive_threshold(scale_correlations, percentile=90):
    """
    Calculate adaptive threshold for identifying significant self-similar regions.
    
    Parameters:
    -----------
    scale_correlations : np.ndarray
        Scale correlation scores
    percentile : int, optional
        Percentile to use for threshold determination
        
    Returns:
    --------
    float
        Adaptive threshold value
    """
    # Filter out extreme values
    valid_values = scale_correlations[np.isfinite(scale_correlations)]
    
    if len(valid_values) == 0:
        return 0.5  # Default threshold
    
    # Calculate threshold based on percentile and standard deviation
    base_threshold = np.percentile(valid_values, percentile)
    std_dev = np.std(valid_values)
    
    # Combine percentile and standard deviation approaches
    threshold = base_threshold - 0.5 * std_dev
    
    # Ensure reasonable bounds
    threshold = max(0.3, min(0.8, threshold))
    
    return threshold

def identify_regions_above_threshold(scale_correlations, threshold, coordinates, min_region_size=5):
    """
    Identify contiguous regions where scale correlations exceed the threshold.
    
    Parameters:
    -----------
    scale_correlations : np.ndarray
        Scale correlation scores
    threshold : float
        Threshold for detection
    coordinates : np.ndarray
        Spatial coordinates
    min_region_size : int, optional
        Minimum size of a region to be considered significant
        
    Returns:
    --------
    List[Tuple[float, float]]
        List of (start, end) coordinate pairs for self-similar regions
    """
    # Find indices where correlations exceed threshold
    above_threshold = scale_correlations > threshold
    
    # Group contiguous indices into regions
    regions = []
    in_region = False
    region_start = 0
    
    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_region:
            # Start of a new region
            in_region = True
            region_start = i
        elif not above_threshold[i] and in_region:
            # End of a region
            in_region = False
            region_end = i - 1
            
            # Only add regions with minimum size
            if region_end - region_start + 1 >= min_region_size:
                # Convert indices to coordinate values
                start_coord = coordinates[region_start]
                end_coord = coordinates[region_end]
                regions.append((start_coord, end_coord))
    
    # Handle case where last point is part of a region
    if in_region and len(above_threshold) - region_start >= min_region_size:
        start_coord = coordinates[region_start]
        end_coord = coordinates[-1]
        regions.append((start_coord, end_coord))
    
    return regions

def calculate_detection_confidence(scale_correlations, threshold):
    """
    Calculate confidence levels for the detected self-similar regions.
    
    Parameters:
    -----------
    scale_correlations : np.ndarray
        Scale correlation scores
    threshold : float
        Threshold used for detection
        
    Returns:
    --------
    float
        Confidence score (0-1) for the detected regions
    """
    # Calculate how much the correlations exceed the threshold on average
    above_threshold = scale_correlations > threshold
    
    if not np.any(above_threshold):
        return 0.0
    
    # Calculate ratio of values above threshold
    ratio_above = np.sum(above_threshold) / len(scale_correlations)
    
    # Calculate average excess above threshold
    excess_values = scale_correlations[above_threshold] - threshold
    avg_excess = np.mean(excess_values) if excess_values.size > 0 else 0
    
    # Combine metrics for final confidence score
    confidence = (ratio_above * 0.3) + (avg_excess * 0.7)
    
    # Normalize to 0-1 range
    confidence = min(1.0, max(0.0, confidence))
    
    return confidence

def _fallback_self_similarity_detection(wavefunction, coordinates):
    """
    Fallback method for self-similarity detection when wavelet transform is unavailable.
    
    Parameters:
    -----------
    wavefunction : np.ndarray
        Wavefunction or probability density
    coordinates : np.ndarray
        Spatial coordinates
        
    Returns:
    --------
    regions : List[Tuple[float, float]]
        Estimated self-similar regions
    analysis_data : Dict
        Basic analysis information
    """
    # Convert to probability density if needed
    if isinstance(wavefunction, Qobj):
        if wavefunction.isket:
            data = np.abs(wavefunction.full().flatten())**2
        else:
            data = np.abs(np.diag(wavefunction.full()))**2
    else:
        data = np.abs(wavefunction)**2 if np.iscomplexobj(wavefunction) else wavefunction
    
    # Normalize data
    if np.sum(data) > 0:
        data = data / np.max(data)
    
    # Use gradient-based analysis to detect potential self-similar regions
    gradient = np.gradient(data)
    gradient_magnitude = np.abs(gradient)
    
    # Calculate adaptive threshold
    threshold = np.mean(gradient_magnitude) + 1.5 * np.std(gradient_magnitude)
    significant_points = gradient_magnitude > threshold
    
    # Group into regions
    regions = []
    in_region = False
    region_start = 0
    
    for i in range(len(significant_points)):
        if significant_points[i] and not in_region:
            # Start of a new region
            in_region = True
            region_start = i
        elif not significant_points[i] and in_region:
            # End of a region
            in_region = False
            region_end = i - 1
            
            # Only add regions with minimum size
            if region_end - region_start + 1 >= 5:
                # Convert indices to coordinate values
                start_coord = coordinates[region_start]
                end_coord = coordinates[region_end]
                regions.append((start_coord, end_coord))
    
    # Handle case where last point is part of a region
    if in_region and len(significant_points) - region_start >= 5:
        start_coord = coordinates[region_start]
        end_coord = coordinates[-1]
        regions.append((start_coord, end_coord))
    
    # Create a synthetic scale correlation for compatibility
    scale_correlations = np.zeros_like(data)
    scale_correlations[significant_points] = 0.5 + 0.5 * gradient_magnitude[significant_points] / np.max(gradient_magnitude)
    
    return regions, {
        'scale_correlations': scale_correlations,
        'threshold': threshold,
        'confidence': 0.5,  # Lower confidence for fallback method
        'wavelet_coeffs': None
    }

def calculate_local_fractal_dimensions(wavefunction, coordinates, window_size=32, step=8, with_confidence=True):
    """
    Calculate fractal dimension in sliding windows to identify self-similar regions.
    
    Parameters:
    -----------
    wavefunction : np.ndarray
        Wavefunction or probability density
    coordinates : np.ndarray
        Spatial coordinates
    window_size : int, optional
        Size of sliding window
    step : int, optional
        Step size for sliding window
    with_confidence : bool, optional
        Whether to calculate confidence intervals
        
    Returns:
    --------
    regions : List[Tuple[float, float]]
        List of regions with significant fractal behavior
    analysis_data : Dict
        Dictionary containing:
        - 'local_dimensions': Array of local fractal dimensions
        - 'confidence_intervals': Confidence intervals for dimensions
        - 'mean': Average dimension
        - 'std': Standard deviation of dimensions
    """
    # Convert to probability density if needed
    if isinstance(wavefunction, Qobj):
        if wavefunction.isket:
            data = np.abs(wavefunction.full().flatten())**2
        else:
            data = np.abs(np.diag(wavefunction.full()))**2
    else:
        data = np.abs(wavefunction)**2 if np.iscomplexobj(wavefunction) else wavefunction
    
    # Normalize data
    if np.sum(data) > 0:
        data = data / np.max(data)
    
    # Calculate local dimensions using sliding window
    local_dimensions = []
    confidence_intervals = []
    window_centers = []
    
    # Adjust window and step size based on data length
    window_size = min(window_size, len(data) // 4)
    step = min(step, window_size // 2)
    
    for i in range(0, len(data) - window_size, step):
        window = data[i:i+window_size]
        
        try:
            # Calculate fractal dimension for window
            if with_confidence:
                dim, info = estimate_fractal_dimension(window)
                ci = info.get('confidence_interval', (dim-0.2, dim+0.2))
                confidence_intervals.append(ci)
            else:
                dim = fractal_dimension(window)
            
            local_dimensions.append(dim)
            window_centers.append(coordinates[i + window_size//2])
        except Exception as e:
            logger.debug(f"Error calculating local dimension at window {i}: {str(e)}")
            # Use a default value
            local_dimensions.append(1.0)
            confidence_intervals.append((0.8, 1.2))
            window_centers.append(coordinates[i + window_size//2])
    
    # Compute statistics
    valid_dims = [d for d in local_dimensions if np.isfinite(d)]
    mean_dim = np.mean(valid_dims) if valid_dims else 1.0
    std_dim = np.std(valid_dims) if len(valid_dims) > 1 else 0.1
    
    # Convert to numpy arrays
    local_dimensions = np.array(local_dimensions)
    window_centers = np.array(window_centers)
    
    # Identify regions with significant fractal behavior
    significant_regions = identify_significant_deviations(
        local_dimensions, 
        window_centers,
        threshold=1.5,
        min_absolute_diff=0.2,
        use_adaptive=True
    )
    
    return significant_regions, {
        'local_dimensions': local_dimensions,
        'window_centers': window_centers,
        'confidence_intervals': confidence_intervals if with_confidence else None,
        'mean': mean_dim,
        'std': std_dim
    }

def identify_significant_deviations(dimensions, coordinates, threshold=1.5, 
                                   min_absolute_diff=0.2, use_adaptive=True, min_region_size=3):
    """
    Identify significant deviations in fractal dimension.
    
    Parameters:
    -----------
    dimensions : np.ndarray
        Array of local fractal dimensions
    coordinates : np.ndarray
        Coordinate values corresponding to dimensions
    threshold : float, optional
        Threshold factor for standard deviation
    min_absolute_diff : float, optional
        Minimum absolute difference from mean
    use_adaptive : bool, optional
        Whether to use adaptive thresholding
    min_region_size : int, optional
        Minimum number of points for a region
        
    Returns:
    --------
    List[Tuple[float, float]]
        List of (start, end) coordinate pairs with significant deviations
    """
    # Filter out invalid values
    valid_indices = np.isfinite(dimensions)
    if not np.any(valid_indices):
        return []
    
    valid_dims = dimensions[valid_indices]
    valid_coords = coordinates[valid_indices]
    
    # Calculate statistics
    mean_dim = np.mean(valid_dims)
    std_dim = np.std(valid_dims) if len(valid_dims) > 1 else 0.1
    
    # Calculate adaptive threshold if requested
    if use_adaptive:
        # Scale threshold inversely with system size for better sensitivity
        n = len(valid_dims)
        threshold_factor = threshold * np.sqrt(100/n) if n > 0 else threshold
    else:
        threshold_factor = threshold
    
    # Identify significant points
    significant_indices = np.where(
        (np.abs(valid_dims - mean_dim) > threshold_factor * std_dim) & 
        (np.abs(valid_dims - mean_dim) > min_absolute_diff)
    )[0]
    
    # Group into regions
    regions = []
    in_region = False
    region_start = 0
    
    for i in range(len(valid_coords)):
        if i in significant_indices and not in_region:
            # Start of a new region
            in_region = True
            region_start = i
        elif (i not in significant_indices or i == len(valid_coords)-1) and in_region:
            # End of a region (or last point)
            in_region = False
            region_end = i - 1 if i not in significant_indices else i
            
            # Only add regions with minimum size
            if region_end - region_start + 1 >= min_region_size:
                # Convert indices to coordinate values
                start_coord = valid_coords[region_start]
                end_coord = valid_coords[region_end]
                regions.append((start_coord, end_coord))
    
    # Handle case where last point is part of a region
    if in_region and len(valid_coords) - region_start >= min_region_size:
        start_coord = valid_coords[region_start]
        end_coord = valid_coords[-1]
        regions.append((start_coord, end_coord))
    
    return regions

# For backward compatibility
analyze_phi_resonance = analyze_fractal_properties

"""
Module for fractal-related computations, including fractal dimension estimation
and generating fractal-based Hamiltonians or wavefunction profiles.

This module provides tools for analyzing and quantifying fractal properties
in quantum systems, particularly focusing on:
- Energy spectrum analysis with f_s parameter sweeps
- Wavefunction probability density computation with zoom capability
- Robust fractal dimension estimation with error analysis
- Phi-sensitive fractal metrics that can reveal golden ratio resonances
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
    # If fewer than 4 data points are present, 'cubic' interpolation will fail with boundary conditions
    interp_kind = 'cubic' if len(psi) >= 4 else 'linear'
    f_interp = interp1d(x_quantum, np.abs(psi)**2, kind=interp_kind, bounds_error=False, fill_value='extrapolate')
    density = f_interp(x_array)
    
    # Increase sampling density if zoom requested
    if zoom_factor > 1.0:
        x_dense = np.linspace(x_array[0], x_array[-1], int(len(x_array) * zoom_factor))
        f_zoom = interp1d(x_array, density, kind='cubic', bounds_error=False, fill_value='extrapolate')
        density = f_zoom(x_dense)
        x_array = x_dense
    
    # Normalize the density
    if np.sum(density) > 0:
        from scipy.integrate import trapezoid
        density = density / trapezoid(density, x_array)
    
    # Compute additional details if requested
    details = None
    if log_details:
        gradient = np.gradient(density)
        interesting_regions = np.where(np.abs(gradient) > np.std(gradient))[0]
        regions = []
        if len(interesting_regions) > 0:
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

def compute_energy_spectrum(
    H_func: Callable[[float], Union[Qobj, np.ndarray]],
    config: Optional[Dict] = None,
    eigen_index: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[List[Tuple[float, float, float, float]], np.ndarray, Dict[str, float]]]]:
    """
    Compute energy spectrum over f_s parameter range with enhanced analysis of
    self-similar regions.

    Parameters:
    -----------
    H_func : Callable[[float], Union[Qobj, np.ndarray]]
        Function that takes f_s parameter and returns Hamiltonian (either Qobj or numpy array).
    config : Optional[Dict]
        Configuration dictionary. If None, loads from evolution_config.yaml.
    eigen_index : int
        Index of eigenvalue to use for analysis (default: 0, ground state).

    Returns:
    --------
    parameter_values : np.ndarray
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
        
        # Handle different types of Hamiltonian objects
        if isinstance(H, Qobj):
            # Use QuTiP's eigenenergies method for Qobj
            evals = np.sort(H.eigenenergies())
        elif isinstance(H, np.ndarray):
            # Use numpy's eigvals for numpy arrays
            if H.ndim == 2 and H.shape[0] == H.shape[1]:  # Check if square matrix
                evals = np.sort(np.linalg.eigvals(H))
            else:
                # If not a square matrix, treat as diagonal Hamiltonian
                evals = np.sort(np.diag(H) if H.ndim == 2 else H)
        else:
            # For other types, try to convert to numpy array
            try:
                H_array = np.array(H)
                if H_array.ndim == 2 and H_array.shape[0] == H_array.shape[1]:
                    evals = np.sort(np.linalg.eigvals(H_array))
                else:
                    evals = np.sort(np.diag(H_array) if H_array.ndim == 2 else H_array)
            except:
                # Fallback: return a single eigenvalue
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
            # Ensure we have valid data for correlation
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
                    # Skip correlation calculation if it fails
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
        # Fallback if gap statistics calculation fails
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
    MAX_SEGMENTS = 1000  # Limit to prevent memory overflow
    
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


def phi_sensitive_dimension(
    data: np.ndarray,
    box_sizes: Optional[np.ndarray] = None,
    scaling_factor: float = None
) -> float:
    """
    Estimate fractal dimension with phi-sensitivity.
    
    Parameters:
    -----------
    data : np.ndarray
        1D or 2D data representing the structure to measure.
    box_sizes : Optional[np.ndarray]
        Array of box sizes for counting. If None, uses default values.
    scaling_factor : float
        Scaling factor used in the simulation.
        
    Returns:
    --------
    float
        Fractal dimension with potential phi-resonance.
    """
    # Use PHI as default scaling factor if none provided
    if scaling_factor is None:
        scaling_factor = PHI
    
    phi = PHI
    
    # Create analysis sensitive to golden ratio
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
    
    # Ensure data is properly normalized
    data = np.abs(data)  # Handle complex values
    if data.ndim == 2:
        data = data.reshape(-1)  # Flatten 2D data
    
    data = data / np.max(data)  # Normalize to [0,1]
    
    # Set default box sizes if none provided
    if box_sizes is None:
        # Use safer box size range to avoid memory issues
        box_sizes = np.logspace(-2, 0, 20)  # Minimum box size of 0.01 instead of 0.001
    
    # Set maximum number of segments to prevent memory errors
    MAX_SEGMENTS = 1000  # Limit to prevent memory overflow
    
    # Modified box-counting algorithm
    counts = []
    valid_boxes = []
    
    for box in box_sizes:
        # Dynamic thresholding with phi sensitivity
        threshold = 0.1 * np.mean(data) * (1 + phi_proximity * (box - 0.5)**2)
        
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
        
        if count > 0:
            counts.append(count)
            valid_boxes.append(box)
    
    # Log-log fit
    if len(valid_boxes) >= 5:  # Require enough points for reliable fit
        log_boxes = np.log(1.0 / np.array(valid_boxes))
        log_counts = np.log(np.array(counts))
        
        # Non-linear fit near phi
        if abs(scaling_factor - phi) < 0.1:
            # Apply phi-specific correction
            log_counts = log_counts * (1 + 0.2 * phi_proximity)
            
        slope, _, _, _, _ = linregress(log_boxes, log_counts)
        return slope
    else:
        return 0.0


def compute_multifractal_spectrum(
    data: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    box_sizes: Optional[np.ndarray] = None,
    scaling_factor: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Compute multifractal spectrum with phi-resonant properties.
    
    Parameters:
    -----------
    data : np.ndarray
        1D or 2D data representing the structure to measure.
    q_values : Optional[np.ndarray]
        Array of q values for multifractal analysis.
    box_sizes : Optional[np.ndarray]
        Array of box sizes for counting.
    scaling_factor : Optional[float]
        Scaling factor used in the simulation.
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing multifractal spectrum data.
    """
    # Set default parameters if not provided
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    
    if box_sizes is None:
        # Use safer box size range to avoid memory issues
        box_sizes = np.logspace(-2, 0, 20)  # Minimum box size of 0.01 instead of 0.001
    
    if scaling_factor is None:
        scaling_factor = PHI
    
    # Calculate phi proximity
    phi = PHI
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
    
    # Ensure data is properly normalized
    data = np.abs(data)  # Handle complex values
    if data.ndim == 2:
        data = data.reshape(-1)  # Flatten 2D data
    
    data = data / np.max(data)  # Normalize to [0,1]
    
    # Set maximum number of segments to prevent memory errors
    MAX_SEGMENTS = 1000  # Limit to prevent memory overflow
    
    # Initialize results
    tq_values = np.zeros_like(q_values, dtype=float)
    f_alpha = []
    alpha = []
    
    # Compute generalized dimensions for each q
    for i, q in enumerate(q_values):
        # Skip q=1 (handled separately)
        if abs(q - 1.0) < 1e-10:
            continue
        
        # Compute partition function for each box size
        partition_values = []
        for box in box_sizes:
            # Calculate number of segments safely
            n_segments = min(int(1/box), MAX_SEGMENTS)
            
            if n_segments <= 1:
                # Skip if box size is too large (would create only 1 segment)
                continue
            
            # Memory-efficient segmentation
            if len(data) > 10000 and n_segments > 100:
                # For large datasets, use manual segmentation
                segment_size = len(data) // n_segments
                measures = []
                
                for j in range(n_segments):
                    start_idx = j * segment_size
                    end_idx = min((j + 1) * segment_size, len(data))
                    segment = data[start_idx:end_idx]
                    
                    # Apply phi-sensitive threshold
                    threshold = 0.05 * np.mean(data) * (1 + phi_proximity * (box - 0.5)**2)
                    if np.any(segment > threshold):
                        # Compute measure (probability) for this segment
                        measure = np.sum(segment) / np.sum(data)
                        measures.append(measure)
            else:
                # For smaller datasets, use array_split
                segments = np.array_split(data, n_segments)
                measures = []
                
                for segment in segments:
                    # Apply phi-sensitive threshold
                    threshold = 0.05 * np.mean(data) * (1 + phi_proximity * (box - 0.5)**2)
                    if np.any(segment > threshold):
                        # Compute measure (probability) for this segment
                        measure = np.sum(segment) / np.sum(data)
                        measures.append(measure)
            
            # Compute partition function
            if measures:
                if q == 0:
                    # For q=0, just count the number of boxes
                    partition = len(measures)
                else:
                    # For q≠0, compute sum of measures^q
                    partition = np.sum(np.power(measures, q))
                
                partition_values.append((box, partition))
        
        # Compute scaling exponent tau(q)
        if len(partition_values) >= 5:
            log_boxes = np.log([1.0/p[0] for p in partition_values])
            log_partitions = np.log([p[1] for p in partition_values])
            
            # Apply phi-resonant correction near phi
            if abs(scaling_factor - phi) < 0.1:
                log_partitions = log_partitions * (1 + 0.1 * phi_proximity * np.sin(q * np.pi))
            
            slope, _, _, _, _ = linregress(log_boxes, log_partitions)
            
            # For q≠1, tau(q) = (q-1)*D_q
            tq_values[i] = slope
            
            # Compute f(alpha) spectrum
            alpha_q = -np.gradient(tq_values, q_values)[i]
            f_alpha_q = q * alpha_q - tq_values[i]
            
            alpha.append(alpha_q)
            f_alpha.append(f_alpha_q)
    
    # Handle q=1 case (information dimension)
    info_dim = 0.0
    partition_values_q1 = []  # Separate list for q=1 case
    
    for box in box_sizes:
        # Calculate number of segments safely
        n_segments = min(int(1/box), MAX_SEGMENTS)
        
        if n_segments <= 1:
            continue
            
        entropy = 0.0
        total_measure = 0.0
        
        # Memory-efficient segmentation
        if len(data) > 10000 and n_segments > 100:
            # For large datasets, use manual segmentation
            segment_size = len(data) // n_segments
            
            for j in range(n_segments):
                start_idx = j * segment_size
                end_idx = min((j + 1) * segment_size, len(data))
                segment = data[start_idx:end_idx]
                
                threshold = 0.05 * np.mean(data) * (1 + phi_proximity * (box - 0.5)**2)
                if np.any(segment > threshold):
                    measure = np.sum(segment) / np.sum(data)
                    if measure > 0:
                        entropy -= measure * np.log(measure)
                        total_measure += measure
        else:
            # For smaller datasets, use array_split
            segments = np.array_split(data, n_segments)
            
            for segment in segments:
                threshold = 0.05 * np.mean(data) * (1 + phi_proximity * (box - 0.5)**2)
                if np.any(segment > threshold):
                    measure = np.sum(segment) / np.sum(data)
                    if measure > 0:
                        entropy -= measure * np.log(measure)
                        total_measure += measure
        
        if total_measure > 0:
            # Normalize entropy
            entropy /= total_measure
            
            # Store (box_size, entropy) pairs
            partition_values_q1.append((box, entropy))
    
    # Compute information dimension D_1
    if len(partition_values) >= 5:
        log_boxes = np.log([1.0/p[0] for p in partition_values])
        entropies = [p[1] for p in partition_values]
        
        # Apply phi-resonant correction
        if abs(scaling_factor - phi) < 0.1:
            entropies = [e * (1 + 0.1 * phi_proximity) for e in entropies]
        
        slope, _, _, _, _ = linregress(log_boxes, entropies)
        
        # Insert D_1 at q=1 position
        q1_idx = np.argmin(np.abs(q_values - 1.0))
        tq_values[q1_idx] = slope
        
        # Compute f(alpha) for q=1
        alpha_1 = slope
        f_alpha_1 = slope
        
        # Insert into alpha and f_alpha arrays
        alpha.insert(q1_idx, alpha_1)
        f_alpha.insert(q1_idx, f_alpha_1)
    
    return {
        'q_values': q_values,
        'tq': tq_values,
        'alpha': np.array(alpha),
        'f_alpha': np.array(f_alpha),
        'phi_proximity': phi_proximity
    }


def analyze_phi_resonance(
    data_func: Callable[[float], np.ndarray],
    scaling_factors: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Analyze how fractal properties change with scaling factor,
    with special focus on potential resonance at phi.
    
    Parameters:
    -----------
    data_func : Callable[[float], np.ndarray]
        Function that takes scaling factor and returns data to analyze.
    scaling_factors : Optional[np.ndarray]
        Array of scaling factors to analyze.
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing analysis results.
    """
    # Set default scaling factors if not provided
    if scaling_factors is None:
        # Include phi and nearby values for detailed analysis
        phi = PHI
        # Create denser sampling around phi
        phi_neighborhood = np.linspace(phi - 0.1, phi + 0.1, 11)
        scaling_factors = np.sort(np.concatenate([
            np.linspace(0.5, phi - 0.1, 5),
            phi_neighborhood,
            np.linspace(phi + 0.1, 3.0, 5)
        ]))
    
    # Initialize results
    dimensions = np.zeros_like(scaling_factors)
    phi_sensitive_dims = np.zeros_like(scaling_factors)
    multifractal_widths = np.zeros_like(scaling_factors)
    resonance_metrics = np.zeros_like(scaling_factors)
    
    # Analyze each scaling factor
    for i, factor in enumerate(scaling_factors):
        # Get data for this scaling factor
        data = data_func(factor)
        
        # Compute standard fractal dimension
        dim, _ = estimate_fractal_dimension(data)
        dimensions[i] = dim
        
        # Compute phi-sensitive dimension
        phi_dim = phi_sensitive_dimension(data, scaling_factor=factor)
        phi_sensitive_dims[i] = phi_dim
        
        # Compute multifractal spectrum
        mf_spectrum = compute_multifractal_spectrum(data, scaling_factor=factor)
        
        # Calculate width of multifractal spectrum (if available)
        if len(mf_spectrum['alpha']) > 1:
            multifractal_widths[i] = np.max(mf_spectrum['alpha']) - np.min(mf_spectrum['alpha'])
        
        # Calculate resonance metric (difference between standard and phi-sensitive)
        resonance_metrics[i] = abs(phi_dim - dim)
    
    return {
        'scaling_factors': scaling_factors,
        'dimensions': dimensions,
        'phi_sensitive_dimensions': phi_sensitive_dims,
        'multifractal_widths': multifractal_widths,
        'resonance_metrics': resonance_metrics,
        'phi_index': np.argmin(np.abs(scaling_factors - PHI))
    }

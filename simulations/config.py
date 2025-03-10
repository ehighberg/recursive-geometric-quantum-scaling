"""
Configuration parameters for quantum simulations with φ-resonant effects.

This module centralizes all configurable parameters used in quantum simulations,
particularly those related to φ (golden ratio) resonance effects. Each parameter
includes detailed documentation about its physical significance, mathematical
justification, and recommended value ranges.
"""

import os
import yaml
from pathlib import Path
import numpy as np
from constants import PHI

# ======= Phi-Resonance Parameters =======

# Width of the Gaussian function used to calculate proximity to φ
# Physical meaning: Controls transition sharpness between φ-resonant and non-resonant behavior
# Mathematical basis: Standard deviation of exp(-(x-μ)²/2σ²) where σ² = PHI_GAUSSIAN_WIDTH/2
# Recommended range: [0.05, 0.2] - Values too small create discontinuous behavior,
#                    values too large blur the distinction of φ-specific effects
PHI_GAUSSIAN_WIDTH = 0.1

# Threshold value for φ-proximity to trigger recursive φ composition
# Physical meaning: Determines when a scaling factor is "close enough" to φ
#                  to exhibit special resonant behavior
# Mathematical basis: Value of exp(-(x-φ)²/PHI_GAUSSIAN_WIDTH) above which 
#                    scaling factors exhibit φ-resonant behaviors
# Recommended range: [0.7, 0.95] - Controls the width of the φ-resonant region
PHI_THRESHOLD = 0.9

# Cutoff value for applying nonlinear correction terms
# Physical meaning: Determines when correction factors are significant enough to apply
# Mathematical basis: Minimum value of the correction factor (φ/(1+|f_s-φ|))
#                    that leads to observable quantum interference effects
# Recommended range: [0.1, 0.3] - Values too low add computation without physical significance,
#                    values too high may miss relevant corrections
CORRECTION_CUTOFF = 0.2

# Parameter controlling transition between GHZ and W states in φ-sensitive states
# Physical meaning: Determines the φ-proximity needed to create maximally entangled states
# Mathematical basis: Threshold for creating GHZ-type entanglement vs W-type entanglement
# Recommended range: [0.7, 0.95] - Higher values create sharper transitions
PHI_WEIGHT_CUTOFF = 0.9

# Parameter for intermediate state blending in φ-sensitive states
# Physical meaning: Proximity to φ at which mixed/intermediate entanglement patterns appear
# Mathematical basis: Secondary threshold for W-state vs product state behavior
# Recommended range: [0.3, 0.7] - Controls the smoothness of the transition away from φ
PHI_INTERMEDIATE_CUTOFF = 0.5

# ======= Numerical Analysis Parameters =======

# Parameters for Savitzky-Golay filter in derivative calculations
# Physical meaning: Controls smoothing of numerical derivatives to reduce noise
# Mathematical basis: Window length and polynomial order for the Savitzky-Golay filter
# Recommended range: Window should be odd, typically 5-11; order should be 2-4
SAVGOL_WINDOW_LENGTH = 5
SAVGOL_POLY_ORDER = 2

# Numerical precision parameters
# Physical meaning: Tolerance thresholds for numerical comparisons
# Mathematical basis: Controls when floating-point values are considered equal
UNITARITY_RTOL = 1e-10  # Relative tolerance for unitarity checks
UNITARITY_ATOL = 1e-10  # Absolute tolerance for unitarity checks

# Default configuration values
_DEFAULT_CONFIG = {
    'total_time': 10.0,
    'n_steps': 50,
    'scale_factor': PHI,
    'recursion_depth': 3,
    'noise': {
        'dephasing': {
            'enabled': False,
            'rate': 0.01
        },
        'amplitude_damping': {
            'enabled': False,
            'rate': 0.01
        }
    }
}

def load_config(config_path=None):
    """
    Load configuration from file or return default config.
    
    Parameters:
    -----------
    config_path : str or Path, optional
        Path to configuration YAML file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Try to load from file if path provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration.")
            return _DEFAULT_CONFIG
    
    # Try to load from default location
    default_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    if os.path.exists(default_path):
        try:
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from default path: {e}")
            print("Using default configuration.")
    
    # Otherwise return default config
    return _DEFAULT_CONFIG.copy()

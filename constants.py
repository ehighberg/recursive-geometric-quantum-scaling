"""Global constants and configuration parameters for Quantum Core."""

# Fractal analysis parameters
import numpy as np

# Phase space grid parameters
GRID_SIZE = 100
XVEC = np.linspace(-10, 10, GRID_SIZE)

# Framework weights for prioritizing analysis components
FRAMEWORK_WEIGHTS = {
    'l1_norm_coherence': 1.0,
    'relative_entropy_coherence': 1.0,
    'robustness_coherence': 1.0,
    'interferometric_power': 1.0,
    'quantum_fisher_information': 1.0,
    'purity': 1.0,
    'entropy': 1.0
}

# Error thresholds for managing validation and handling anomalies in analyses
ERROR_THRESHOLDS = {
    'numerical': 0.001,  # Numerical precision threshold
}

# Additional constants can be added below

# Phi-based recursion parameters
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
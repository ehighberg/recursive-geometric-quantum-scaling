"""
Fixed implementation of scaled unitary operators using qutip's features.

This module provides corrected implementations for:
- Linear scaling of unitary operators
- Recursive geometric scaling with golden ratio properties

The implementations ensure consistent scaling factor application without
artificially enhancing phi-related effects.
"""

import numpy as np
from qutip import Qobj, propagator, sigmax, sigmaz, qeye
from qutip.solver import Result
from qutip_qip.operations import Gate
from qutip_qip.circuit import QubitCircuit
from constants import PHI
from simulations.config import (
    PHI_GAUSSIAN_WIDTH, PHI_THRESHOLD, CORRECTION_CUTOFF,
    UNITARITY_RTOL, UNITARITY_ATOL
)

def _ensure_unitarity(U):
    """
    Helper function to ensure a matrix is unitary by performing SVD normalization.
    
    Parameters:
    -----------
    U (Qobj): Input operator to normalize
    
    Returns:
    --------
    Qobj: Unitarized operator
    """
    # Perform singular value decomposition
    u, s, vh = np.linalg.svd(U.full())
    
    # Reconstruct with all singular values = 1
    unitarized = np.dot(u, vh)
    
    # Return as Qobj with same dimensions
    return Qobj(unitarized, dims=U.dims)

def get_scaled_unitary_fixed(H, time, scaling_factor=1.0):
    """
    Get the linearly scaled unitary operator for a given Hamiltonian.
    
    This uses matrix logarithm and exponentiation to perform continuous scaling
    of a unitary operator: U^s = exp(s * log(U)) where s is scaling_factor.
    
    Parameters:
    -----------
    H (Qobj): Hamiltonian operator (unscaled)
    time (float): Evolution time
    scaling_factor (float): Factor to scale the unitary (applied ONCE)
    
    Returns:
    --------
    Qobj: Scaled unitary operator
    """
    # Special case: scaling_factor = 1.0 returns the standard unitary
    if scaling_factor == 1.0:
        # Get unitary for unscaled Hamiltonian
        return propagator(H, time)
    
    # For other scaling factors, create a scaled Hamiltonian
    # Apply scaling factor ONCE here
    H_scaled = scaling_factor * H
    
    # Get unitary for scaled Hamiltonian
    U = propagator(H_scaled, time)
    
    # Ensure the result preserves unitarity (within numerical precision)
    unitarity_check = (U * U.dag())
    dims = unitarity_check.dims
    I = Qobj(np.eye(U.shape[0]), dims=dims)
    if not np.allclose((U * U.dag()).full(), I.full(), 
                       rtol=UNITARITY_RTOL, atol=UNITARITY_ATOL):
        # Apply re-normalization to ensure unitarity
        U = _ensure_unitarity(U)
    
    return U

def get_phi_recursive_unitary_fixed(H, time, scaling_factor=1.0, recursion_depth=3):
    """
    Generate a phi-recursive unitary operator with consistent parameter usage.
    
    Mathematical model:
    U_φ(t) = U(t/φ) · U(t/φ²)
    
    where U(t) = exp(-i·H·t·scaling_factor)
    
    Parameters:
    -----------
    H : Qobj
        Hamiltonian operator (unscaled)
    time : float
        Evolution time
    scaling_factor : float
        Scaling factor for the Hamiltonian (applied consistently at each level)
    recursion_depth : int
        Recursion depth (0 means no recursion)
        
    Returns:
    --------
    Qobj: Unitary evolution operator
    """
    # Base case: no recursion or invalid recursion depth
    if recursion_depth <= 0:
        # Apply standard time evolution with scaling_factor
        # The scaling factor is applied ONCE here
        H_scaled = scaling_factor * H
        return (-1j * H_scaled * time).expm()
    
    # Recursive case: implement the mathematical relation U_φ(t) = U(t/φ) · U(t/φ²)
    # Apply recursion with proper parameter passing:
    # - Pass the SAME scaling_factor down the recursion chain
    # - Only modify the time parameter with phi divisions
    U_phi1 = get_phi_recursive_unitary_fixed(H, time/PHI, scaling_factor, recursion_depth-1)
    U_phi2 = get_phi_recursive_unitary_fixed(H, time/(PHI**2), scaling_factor, recursion_depth-1)
    
    # Combine recursive unitaries
    return U_phi1 * U_phi2
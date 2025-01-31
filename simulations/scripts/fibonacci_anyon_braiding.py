#!/usr/bin/env python
# simulations/scripts/fibonacci_anyon_braiding.py

"""
Fibonacci Anyon Braiding Operations

This module defines the braiding operators for Fibonacci anyons in a 2D subspace.
Using the fusion rules τ × τ = 1 + τ, the Hilbert space of three anyons (with total charge τ)
is two-dimensional. Braiding operators are constructed using:
  - The F-matrix: F = [[φ⁻¹, φ^(-1/2)],
                         [φ^(-1/2), -φ⁻¹]]
  - The R-matrices:
         R₁ = exp(-4πi/5) and R_τ = exp(3πi/5)
These ingredients yield elementary braiding operations that form a unitary representation
of the braid group for Fibonacci anyons.
"""

import numpy as np
import math
from qutip import Qobj

class BraidGenerator:
    """
    Generalized braiding operator generator for Fibonacci anyons.
    
    This class calculates the F-matrix and elementary braiding operators for a 2D representation.
    """
    def __init__(self, dimensionality=2):
        if dimensionality != 2:
            raise NotImplementedError("Only 2D implementation is provided for Fibonacci anyons.")
        self.dim = dimensionality
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def F_matrix(self):
        """
        Compute the F-matrix for Fibonacci anyons in 2D.
        
        Returns:
            Qobj: The 2x2 F-matrix defined as:
                 [[φ⁻¹, φ^(-1/2)],
                  [φ^(-1/2), -φ⁻¹]]
        """
        phi_inv = 1.0 / self.phi
        phi_inv_sqrt = self.phi ** (-0.5)
        mat = np.array([
            [phi_inv,      phi_inv_sqrt],
            [phi_inv_sqrt, -phi_inv]
        ], dtype=complex)
        return Qobj(mat)

    def elementary_braid(self, braid_index: int):
        """
        Compute the elementary braid operator for the specified braid index.
        
        For braid_index = 1, returns:
            B₁ = diag(exp(-4πi/5), exp(3πi/5))
        For braid_index = 2, returns:
            B₂ = F⁻¹ B₁ F
        """
        if braid_index == 1:
            R_1   = np.exp(-4j * math.pi / 5)
            R_tau = np.exp(3j * math.pi / 5)
            return Qobj(np.diag([R_1, R_tau]))
        elif braid_index == 2:
            F = self.F_matrix()
            return F.inv() * self.elementary_braid(1) * F
        else:
            raise ValueError("Unsupported braid index for Fibonacci anyons. Use 1 or 2.")

# Preserve legacy API:
_default_braid_gen = BraidGenerator(dimensionality=2)

def F_2x2():
    """Return the 2x2 F-matrix for Fibonacci anyons."""
    return _default_braid_gen.F_matrix()

def F_inv_2x2():
    """Return the inverse of the 2x2 F-matrix for Fibonacci anyons."""
    return _default_braid_gen.F_matrix().inv()

def braid_b1_2d():
    """Return the elementary braid operator for braid index 1 in 2D."""
    return _default_braid_gen.elementary_braid(1)

def braid_b2_2d():
    """Return the elementary braid operator for braid index 2 in 2D."""
    return _default_braid_gen.elementary_braid(2)

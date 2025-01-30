#!/usr/bin/env python
# simulations/scripts/fibonacci_anyon_braiding.py

"""
B1, B2 braiding ops in 2D for 3 Fibonacci anyons w/ total charge tau.
We define F_2x2, its inverse, and the phases R_1, R_tau.
"""

import numpy as np
import math
from qutip import Qobj

class BraidGenerator:
    """Generalized braiding operator generator for Fibonacci anyons"""
    def __init__(self, dimensionality=2):
        self.dim = dimensionality
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def F_matrix(self):
        """F-matrix for current dimensionality"""
        if self.dim == 2:
            mat = (1.0/np.sqrt(self.phi))*np.array([
                [1,         np.sqrt(self.phi)],
                [np.sqrt(self.phi),   -1     ]
            ], dtype=complex)
            return Qobj(mat)
        raise NotImplementedError("Higher dimensions require topological data")

    def elementary_braid(self, braid_index: int):
        """General braid operator for index position"""
        if self.dim == 2:
            if braid_index == 1:
                R_1   = np.exp(-4j*math.pi/5)
                R_tau = np.exp( 3j*math.pi/5)
                return Qobj(np.diag([R_1, R_tau]))
            elif braid_index == 2:
                return self.F_matrix().inv() * self.elementary_braid(1) * self.F_matrix()
        raise NotImplementedError("Higher dimensional braids not implemented")

# Preserve existing API with default 2D generator
_default_braid_gen = BraidGenerator(dimensionality=2)

def F_2x2():
    """Legacy 2D F-matrix (now wraps BraidGenerator)"""
    return _default_braid_gen.F_matrix()

def F_inv_2x2():
    """Legacy 2D inverse F-matrix"""
    return _default_braid_gen.F_matrix().inv()

def braid_b1_2d():
    """Legacy 2D B1 braid"""
    return _default_braid_gen.elementary_braid(1)

def braid_b2_2d():
    """Legacy 2D B2 braid"""
    return _default_braid_gen.elementary_braid(2)

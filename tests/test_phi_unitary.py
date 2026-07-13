"""
Comprehensive test suite for validating unitarity and mathematical 
properties of scaled and phi-recursive unitary operators.

These tests ensure that unitary operators maintain their mathematical 
properties across different scaling regimes, particularly around phi.
"""

import pytest
import numpy as np
from qutip import sigmax, sigmay, sigmaz, tensor, qeye, Qobj
from simulations.scaled_unitary import get_scaled_unitary, get_phi_recursive_unitary
from constants import PHI
from simulations.config import (
    PHI_GAUSSIAN_WIDTH, PHI_THRESHOLD, CORRECTION_CUTOFF,
    UNITARITY_RTOL, UNITARITY_ATOL
)

# Create test Hamiltonians with different properties
@pytest.fixture
def test_hamiltonians():
    """Provide a set of test Hamiltonians with different properties."""
    return {
        "simple": 0.5 * sigmax(),
        "mixed": sigmaz() + 0.5 * sigmax(),
        "two_qubit": tensor(sigmax(), sigmaz()) + tensor(sigmaz(), sigmax()),
        "three_qubit": (
            tensor(sigmax(), qeye(2), qeye(2)) + 
            tensor(qeye(2), sigmay(), qeye(2)) + 
            tensor(qeye(2), qeye(2), sigmaz())
        )
    }

# Define test scaling factors including phi and values near phi
@pytest.fixture
def scaling_factors():
    """Provide a range of scaling factors including phi and values near it."""
    # Include phi, values near phi, and some distant values
    return [0.5, 1.0, 1.2, 1.5, PHI-0.01, PHI, PHI+0.01, 1.7, 2.0, 3.0]

# Test that scaled unitaries maintain unitarity
def test_scaled_unitary_maintains_unitarity(test_hamiltonians, scaling_factors):
    """Test that get_scaled_unitary maintains unitarity for all scaling factors."""
    time = 1.0
    for name, H in test_hamiltonians.items():
        for factor in scaling_factors:
            U = get_scaled_unitary(H, time, factor)
            
            # Check that U is unitary: U * U^† = I
            I = qeye(H.dims[0])
            U_product = U * U.dag()
            
            # Calculate error
            error = (U_product - I).norm()
            
            # Assert with detailed message
            assert error < 1e-10, (
                f"Unitarity violated for H={name}, f_s={factor}: "
                f"||U*U^† - I|| = {error} > 1e-10"
            )

# Test that phi-recursive unitaries maintain unitarity
def test_phi_recursive_unitary_maintains_unitarity(test_hamiltonians, scaling_factors):
    """Test that get_phi_recursive_unitary maintains unitarity across scaling factors and depths."""
    time = 1.0
    recursion_depths = [0, 1, 2, 3]
    
    for name, H in test_hamiltonians.items():
        for factor in scaling_factors:
            for depth in recursion_depths:
                U = get_phi_recursive_unitary(H, time, factor, depth)
                
                # Check that U is unitary: U * U^† = I
                I = qeye(H.dims[0])
                U_product = U * U.dag()
                
                # Calculate error with detailed reporting
                error = (U_product - I).norm()
                
                # Assert with detailed message
                assert error < 1e-10, (
                    f"Unitarity violated for H={name}, f_s={factor}, depth={depth}: "
                    f"||U*U^† - I|| = {error} > 1e-10"
                )

# Test that eigenvalues maintain unit modulus
def test_eigenvalues_have_unit_modulus(test_hamiltonians, scaling_factors):
    """Test that eigenvalues of unitaries have unit modulus."""
    time = 1.0
    
    for name, H in test_hamiltonians.items():
        for factor in scaling_factors:
            # Get unitary operators
            U_standard = get_scaled_unitary(H, time, factor)
            U_phi = get_phi_recursive_unitary(H, time, factor, recursion_depth=2)
            
            # Check eigenvalues for standard unitary
            eigvals_standard = U_standard.eigenenergies()
            for val in eigvals_standard:
                assert abs(abs(val) - 1.0) < 1e-10, (
                    f"Standard unitary eigenvalue {val} has modulus {abs(val)} ≠ 1"
                )
            
            # Check eigenvalues for phi recursive unitary
            eigvals_phi = U_phi.eigenenergies()
            for val in eigvals_phi:
                assert abs(abs(val) - 1.0) < 1e-10, (
                    f"Phi recursive unitary eigenvalue {val} has modulus {abs(val)} ≠ 1"
                )

# Test consistency between scaled unitaries and recursive unitaries
def test_recursive_identity(test_hamiltonians):
    """The documented recurrence is algebraically equal to standard evolution."""
    time = 1.0
    
    # Define scaling factors including phi and non-phi values
    scaling_factors = [1.0, 1.5, PHI, 1.7, 2.0]
    
    for name, H in test_hamiltonians.items():
        # Calculate unitaries for each scaling factor
        phi_recursive_unitaries = {}
        standard_unitaries = {}
        
        for factor in scaling_factors:
            phi_recursive_unitaries[factor] = get_phi_recursive_unitary(H, time, factor, recursion_depth=3)
            standard_unitaries[factor] = get_scaled_unitary(H, time, factor)
        
        # 1/phi + 1/phi^2 == 1. Every factor is a function of the same
        # Hamiltonian, so the recursive exponentials commute and compose to U(t).
        for factor in scaling_factors:
            difference = (
                phi_recursive_unitaries[factor] - standard_unitaries[factor]
            ).norm()
            assert difference < 1e-10, (
                f"Recursive identity drifted at f_s={factor}: {difference}"
            )

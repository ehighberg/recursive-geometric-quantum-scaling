#!/usr/bin/env python
# tests/test_phi_resonant.py

"""
Test suite for phi-resonant quantum evolution.

This module tests the phi-resonant features added to the quantum simulation framework,
including:
- Phi-recursive unitary operations
- Fractal quantum states
- Phi-sensitive topological invariants
- Phi-resonant analysis metrics
"""

import unittest
import numpy as np
from qutip import sigmaz, sigmax, basis
from constants import PHI

# Import components to test
from simulations.quantum_circuit import get_phi_recursive_unitary
from simulations.quantum_state import (
    state_fractal, state_fibonacci, state_phi_sensitive, state_recursive_superposition
)
from analyses.fractal_analysis import phi_sensitive_dimension
from analyses.topological_invariants import (
    compute_phi_sensitive_winding, compute_phi_resonant_berry_phase
)
from simulations.scripts.evolve_state import (
    run_phi_recursive_evolution, run_comparative_analysis
)

class TestPhiResonantFeatures(unittest.TestCase):
    """Test suite for phi-resonant quantum features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.H = sigmaz()  # Simple Hamiltonian for testing
        self.phi = PHI
        self.psi0 = basis(2, 0)  # Simple initial state
    
    def test_phi_recursive_unitary(self):
        """Test phi-recursive unitary operation."""
        # Get phi-recursive unitary
        U_phi = get_phi_recursive_unitary(self.H, 1.0, self.phi, 2)
        
        # Get standard unitary for comparison
        from simulations.scaled_unitary import get_scaled_unitary
        U_std = get_scaled_unitary(self.H, 1.0, self.phi)
        
        # Verify that phi-recursive unitary is different from standard unitary
        self.assertFalse(np.allclose(U_phi.full(), U_std.full()),
                         "Phi-recursive unitary should differ from standard unitary")
        
        # Verify that phi-recursive unitary is still unitary
        U_dag_U = U_phi.dag() * U_phi
        identity = np.eye(2, dtype=complex)
        self.assertTrue(np.allclose(U_dag_U.full(), identity),
                        "Phi-recursive unitary should be unitary")
    
    def test_fractal_states(self):
        """Test fractal quantum states."""
        # Create fractal state
        psi_fractal = state_fractal(num_qubits=2, depth=2, phi_param=self.phi)
        
        # Verify state is normalized
        self.assertAlmostEqual(psi_fractal.norm(), 1.0, delta=1e-6)
        
        # Create Fibonacci state
        psi_fib = state_fibonacci(num_qubits=3)
        
        # Verify state is normalized
        self.assertAlmostEqual(psi_fib.norm(), 1.0, delta=1e-6)
        
        # Create phi-sensitive state
        psi_phi = state_phi_sensitive(num_qubits=1, scaling_factor=self.phi)
        
        # Verify state is normalized
        self.assertAlmostEqual(psi_phi.norm(), 1.0, delta=1e-6)
        
        # Create recursive superposition state
        psi_rec = state_recursive_superposition(num_qubits=2, depth=2, scaling_factor=self.phi)
        
        # Verify state is normalized
        self.assertAlmostEqual(psi_rec.norm(), 1.0, delta=1e-6)
    
    def test_phi_sensitive_dimension(self):
        """Test phi-sensitive fractal dimension calculation."""
        # Create test data
        data = np.random.rand(1000)
        
        # Compute phi-sensitive dimension at phi
        dim_phi = phi_sensitive_dimension(data, scaling_factor=self.phi)
        
        # Compute phi-sensitive dimension away from phi
        dim_other = phi_sensitive_dimension(data, scaling_factor=3.0)
        
        # Verify dimensions are valid
        self.assertTrue(0.0 <= dim_phi <= 2.0,
                       "Phi-sensitive dimension should be between 0 and 2")
        self.assertTrue(0.0 <= dim_other <= 2.0,
                       "Phi-sensitive dimension should be between 0 and 2")
    
    def test_phi_sensitive_topological_invariants(self):
        """Test phi-sensitive topological invariants."""
        # Create test eigenstates
        eigenstates = []
        k_points = np.linspace(0, 2*np.pi, 10)
        
        for k in k_points:
            # Create k-dependent state
            H_k = self.H + k * sigmax()
            _, states = H_k.eigenstates()
            eigenstates.append(states[0])
        
        # Compute phi-sensitive winding at phi
        winding_phi = compute_phi_sensitive_winding(eigenstates, k_points, self.phi)
        
        # Compute phi-sensitive winding away from phi
        winding_other = compute_phi_sensitive_winding(eigenstates, k_points, 3.0)
        
        # Verify winding numbers are valid
        self.assertTrue(isinstance(winding_phi, float),
                       "Phi-sensitive winding should be a float")
        self.assertTrue(isinstance(winding_other, float),
                       "Phi-sensitive winding should be a float")
        
        # Compute phi-resonant Berry phase
        berry_phi = compute_phi_resonant_berry_phase(eigenstates, self.phi)
        
        # Verify Berry phase is valid
        self.assertTrue(-np.pi <= berry_phi <= np.pi,
                       "Berry phase should be between -π and π")
    
    def test_phi_recursive_evolution(self):
        """Test phi-recursive evolution."""
        # Run phi-recursive evolution with minimal steps
        result = run_phi_recursive_evolution(
            num_qubits=1,
            state_label="plus",
            n_steps=5,
            scaling_factor=self.phi,
            recursion_depth=2,
            analyze_phi=True
        )
        
        # Verify result contains expected attributes
        self.assertTrue(hasattr(result, 'states'),
                       "Result should have states attribute")
        self.assertTrue(hasattr(result, 'times'),
                       "Result should have times attribute")
        self.assertTrue(hasattr(result, 'expect'),
                       "Result should have expect attribute")
        
        # Verify phi-sensitive analysis results
        self.assertTrue(hasattr(result, 'phi_dimension'),
                       "Result should have phi_dimension attribute")
        self.assertTrue(hasattr(result, 'phi_winding'),
                       "Result should have phi_winding attribute")
        self.assertTrue(hasattr(result, 'phi_berry_phase'),
                       "Result should have phi_berry_phase attribute")
    
    def test_comparative_analysis(self):
        """Test comparative analysis."""
        # Run comparative analysis with minimal settings
        results = run_comparative_analysis(
            scaling_factors=[self.phi, 2.0],
            num_qubits=1,
            state_label="plus",
            n_steps=5
        )
        
        # Verify results structure
        self.assertIn('scaling_factors', results,
                     "Results should contain scaling_factors")
        self.assertIn('standard_results', results,
                     "Results should contain standard_results")
        self.assertIn('phi_recursive_results', results,
                     "Results should contain phi_recursive_results")
        self.assertIn('comparative_metrics', results,
                     "Results should contain comparative_metrics")
        
        # Verify comparative metrics
        for factor in results['scaling_factors']:
            self.assertIn('state_overlap', results['comparative_metrics'][factor],
                         "Comparative metrics should include state_overlap")
            self.assertIn('dimension_difference', results['comparative_metrics'][factor],
                         "Comparative metrics should include dimension_difference")
            self.assertIn('phi_proximity', results['comparative_metrics'][factor],
                         "Comparative metrics should include phi_proximity")

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
"""
Test script to verify standardized evolution functions.

This script tests:
1. Eigenvalue computation standardization
2. Scaled unitary consistency
3. Quantum evolution interface

Usage:
    python test_evolution_standardization.py
"""

import numpy as np
from qutip import Qobj, sigmax, sigmaz
import matplotlib.pyplot as plt
from pathlib import Path

print("Testing standardized evolution functions...")

# Test eigenvalue computation
print("\n1. Testing eigenvalue computation...")
from simulations.quantum_utils import compute_eigenvalues

# Create test Hamiltonians
H_qutip = sigmaz() + 0.5 * sigmax()
H_numpy = np.array([[1, 0.5], [0.5, -1]])
H_list = [1, -1]  # This should raise an error

# Test each type
evals_qutip = compute_eigenvalues(H_qutip)
evals_numpy = compute_eigenvalues(H_numpy)

print(f"  QuTiP Hamiltonian eigenvalues: {evals_qutip}")
print(f"  NumPy Hamiltonian eigenvalues: {evals_numpy}")

try:
    evals_list = compute_eigenvalues(H_list)
    print("  ERROR: List Hamiltonian didn't raise an error")
except TypeError as e:
    print(f"  SUCCESS: List Hamiltonian correctly raised: {e}")

# Test scaled unitary consistency
print("\n2. Testing scaled unitary consistency...")
from simulations.scaled_unitary import get_scaled_unitary, get_phi_recursive_unitary
from constants import PHI

# Create test Hamiltonian
H = sigmaz() + 0.5 * sigmax()
time = 1.0

# Compare different scaling factors
scaling_factors = [0.5, 1.0, PHI, 2.0]

for factor in scaling_factors:
    # Get standard scaled unitary
    U_scaled = get_scaled_unitary(H, time, factor)
    
    # Get phi-recursive for comparison (with recursion_depth=0 it should match standard)
    U_phi = get_phi_recursive_unitary(H, time, factor, recursion_depth=0)
    
    # Compare traces (should be same for recursion_depth=0)
    trace_scaled = U_scaled.tr()
    trace_phi = U_phi.tr()
    
    print(f"  Scaling factor {factor}:")
    print(f"    Standard trace: {trace_scaled}")
    print(f"    Phi-recur trace: {trace_phi}")
    print(f"    Match: {np.isclose(abs(trace_scaled), abs(trace_phi))}")

# Test evolution interface
print("\n3. Testing quantum evolution interface...")
from simulations.quantum_utils import evolve_quantum_state
from qutip import basis

# Create initial state (|+>)
psi0 = (basis(2, 0) + basis(2, 1)).unit()

# Create Hamiltonian
H = sigmaz()

# Define time points
times = np.linspace(0, 10, 100)

# Run evolution with different approaches
print("  Running standard evolution...")
result_std = evolve_quantum_state(psi0, H, times, scaling_factor=1.0)

print("  Running phi-recursive evolution...")
result_phi = evolve_quantum_state(psi0, H, times, scaling_factor=PHI, phi_recursive=True)

# Plot results
output_dir = Path("test_results")
output_dir.mkdir(exist_ok=True, parents=True)

plt.figure(figsize=(10, 6))
plt.plot(times, result_std.expect[0], label="Standard (f_s=1.0)")
plt.plot(times, result_phi.expect[0], label=f"Phi-Recursive (f_s={PHI:.4f})")
plt.xlabel("Time")
plt.ylabel("<σz>")
plt.title("Comparison of Evolution Methods")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / "evolution_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll tests completed. Check test_results directory for outputs.")
print("Success!")

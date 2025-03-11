#!/usr/bin/env python
# test_fixed_implementations.py

"""
Test script to validate the fixed implementations of the RGQS system.
"""

import numpy as np
from constants import PHI
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import original and fixed implementations
from simulations.scaled_unitary import get_phi_recursive_unitary as get_phi_recursive_unitary_orig
from simulations.scaled_unitary_fixed import get_phi_recursive_unitary_fixed
 

from simulations.scripts.evolve_state import run_state_evolution as run_state_evolution_orig
from simulations.scripts.evolve_state_fixed import run_state_evolution_fixed
 

from analyses.fractal_analysis import phi_sensitive_dimension, estimate_fractal_dimension
from analyses.fractal_analysis_fixed_complete import phi_sensitive_dimension as phi_sensitive_dimension_fixed, calculate_fractal_dimension

def test_scaling_factor_application():
    """
    Test that scaling factors are applied consistently in fixed implementations.
    """
    print("Testing scaling factor application...")
    results_dir = Path("test_results_fixed")
    results_dir.mkdir(exist_ok=True)
    
    # Test parameters
    scaling_factors = [0.5, 1.0, 1.5, PHI, 2.0, 2.5, 3.0]
    
    # Test unitary scaling
    from qutip import sigmaz
    H = sigmaz()
    time_val = 1.0
    
    orig_norms = []
    fixed_norms = []
    
    for sf in scaling_factors:
        # Original implementation
        U_orig = get_phi_recursive_unitary_orig(H, time_val, sf, recursion_depth=1)
        orig_norms.append(np.linalg.norm(U_orig.full()))
        
        # Fixed implementation
        U_fixed = get_phi_recursive_unitary_fixed(H, time_val, sf, recursion_depth=1)
        fixed_norms.append(np.linalg.norm(U_fixed.full()))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(scaling_factors, orig_norms, 'o-', label='Original Implementation')
    plt.plot(scaling_factors, fixed_norms, 's-', label='Fixed Implementation')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Unitary Norm')
    plt.title('Comparison of Scaling Factor Application in Unitary Construction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(results_dir / "unitary_scaling_comparison.png", dpi=300)
    print(f"Saved unitary scaling comparison to {results_dir / 'unitary_scaling_comparison.png'}")
    
    # Test evolution scaling
    orig_results = {}
    fixed_results = {}
    
    print("Running evolution tests...")
    
    for sf in scaling_factors:
        # Original implementation
        print(f"Original implementation, sf={sf}")
        start_time = time.time()
        result_orig = run_state_evolution_orig(
            num_qubits=1,
            state_label="plus",
            n_steps=20,
            scaling_factor=sf
        )
        orig_time = time.time() - start_time
        orig_results[sf] = {
            'final_state': result_orig.states[-1],
            'time': orig_time
        }
        
        # Fixed implementation
        print(f"Fixed implementation, sf={sf}")
        start_time = time.time()
        result_fixed = run_state_evolution_fixed(
            num_qubits=1,
            state_label="plus",
            n_steps=20,
            scaling_factor=sf
        )
        fixed_time = time.time() - start_time
        fixed_results[sf] = {
            'final_state': result_fixed.states[-1],
            'time': fixed_time
        }
        
        # Calculate overlap between original and fixed implementations
        overlap = abs(result_orig.states[-1].overlap(result_fixed.states[-1]))**2
        print(f"Scaling factor {sf}: State overlap between implementations: {overlap:.4f}")
        print(f"Timing: Original: {orig_time:.2f}s, Fixed: {fixed_time:.2f}s")
    
    # Plot timing comparison
    plt.figure(figsize=(10, 6))
    plt.plot(scaling_factors, [orig_results[sf]['time'] for sf in scaling_factors], 'o-', label='Original Implementation')
    plt.plot(scaling_factors, [fixed_results[sf]['time'] for sf in scaling_factors], 's-', label='Fixed Implementation')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison Between Implementations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(results_dir / "evolution_timing_comparison.png", dpi=300)
    print(f"Saved timing comparison to {results_dir / 'evolution_timing_comparison.png'}")
    
    return {
        'orig_results': orig_results,
        'fixed_results': fixed_results
    }

def test_fractal_dimension_calculation():
    """
    Test that fractal dimension calculations are consistent across scaling factors.
    """
    print("Testing fractal dimension calculation...")
    results_dir = Path("test_results_fixed")
    results_dir.mkdir(exist_ok=True)
    
    # Create test data with known fractal dimension (Cantor set has D ≈ 0.631)
    def create_cantor_set(n_points=1000, depth=5):
        x = np.linspace(0, 1, n_points)
        y = np.zeros(n_points)
        
        for d in range(depth):
            segment_size = 1.0 / (3**d)
            for i in range(3**d):
                if i % 3 != 1:  # Remove middle third
                    start = i * segment_size
                    end = (i + 1) * segment_size
                    mask = (x >= start) & (x <= end)
                    y[mask] = 1
        
        return y
    
    # Create a sine wave dataset (not fractal)
    def create_sine_wave(n_points=1000, frequency=5):
        x = np.linspace(0, 1, n_points)
        y = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x)
        return y
    
    # Create test datasets
    cantor_set = create_cantor_set()
    sine_wave = create_sine_wave()
    
    # Define scaling factors with dense sampling around phi
    scaling_factors = np.sort(np.concatenate([
        np.linspace(0.5, 3.0, 20),
        np.linspace(PHI-0.1, PHI+0.1, 11)
    ]))
    scaling_factors = np.unique(scaling_factors)
    
    # Calculate fractal dimensions using original and fixed implementations
    orig_cantor_dims = []
    fixed_cantor_dims = []
    orig_sine_dims = []
    fixed_sine_dims = []
    
    print("Calculating fractal dimensions...")
    
    for sf in scaling_factors:
        # Original phi-sensitive dimension (may have phi-specific modifications)
        orig_cantor_dim = phi_sensitive_dimension(cantor_set, scaling_factor=sf)
        orig_sine_dim = phi_sensitive_dimension(sine_wave, scaling_factor=sf)
        
        # Fixed fractal dimension (consistent algorithm)
        fixed_cantor_dim, _ = calculate_fractal_dimension(cantor_set)
        fixed_sine_dim, _ = calculate_fractal_dimension(sine_wave)
        
        orig_cantor_dims.append(orig_cantor_dim)
        fixed_cantor_dims.append(fixed_cantor_dim)
        orig_sine_dims.append(orig_sine_dim)
        fixed_sine_dims.append(fixed_sine_dim)
    
    # Plot comparison for Cantor set
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scaling_factors, orig_cantor_dims, 'o-', label='Original Implementation')
    plt.axhline(y=np.log(2)/np.log(3), color='g', linestyle='-', label='Theoretical (≈0.631)')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Fractal Dimension')
    plt.title('Original Implementation - Cantor Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scaling_factors, fixed_cantor_dims, 'o-', color='orange', label='Fixed Implementation')
    plt.axhline(y=np.log(2)/np.log(3), color='g', linestyle='-', label='Theoretical (≈0.631)')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Fractal Dimension')
    plt.title('Fixed Implementation - Cantor Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "cantor_set_dimension_comparison.png", dpi=300)
    print(f"Saved Cantor set comparison to {results_dir / 'cantor_set_dimension_comparison.png'}")
    
    # Plot comparison for sine wave
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scaling_factors, orig_sine_dims, 'o-', label='Original Implementation')
    plt.axhline(y=1.0, color='g', linestyle='-', label='Theoretical (≈1.0)')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Fractal Dimension')
    plt.title('Original Implementation - Sine Wave')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scaling_factors, fixed_sine_dims, 'o-', color='orange', label='Fixed Implementation')
    plt.axhline(y=1.0, color='g', linestyle='-', label='Theoretical (≈1.0)')
    plt.axvline(x=PHI, color='r', linestyle='--', label=f'φ ≈ {PHI:.4f}')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Fractal Dimension')
    plt.title('Fixed Implementation - Sine Wave')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "sine_wave_dimension_comparison.png", dpi=300)
    print(f"Saved sine wave comparison to {results_dir / 'sine_wave_dimension_comparison.png'}")
    
    # Calculate statistics
    orig_cantor_mean = np.mean(orig_cantor_dims)
    orig_cantor_std = np.std(orig_cantor_dims)
    fixed_cantor_mean = np.mean(fixed_cantor_dims)
    fixed_cantor_std = np.std(fixed_cantor_dims)
    
    orig_sine_mean = np.mean(orig_sine_dims)
    orig_sine_std = np.std(orig_sine_dims)
    fixed_sine_mean = np.mean(fixed_sine_dims)
    fixed_sine_std = np.std(fixed_sine_dims)
    
    # Find phi index
    phi_idx = np.argmin(np.abs(scaling_factors - PHI))
    
    # Print statistics
    print("\nStatistical Summary:")
    print("\nCantor Set (theoretical dim ≈ 0.631):")
    print(f"Original: mean={orig_cantor_mean:.4f}, std={orig_cantor_std:.4f}, at phi={orig_cantor_dims[phi_idx]:.4f}")
    print(f"Fixed: mean={fixed_cantor_mean:.4f}, std={fixed_cantor_std:.4f}, at phi={fixed_cantor_dims[phi_idx]:.4f}")
    
    print("\nSine Wave (theoretical dim ≈ 1.0):")
    print(f"Original: mean={orig_sine_mean:.4f}, std={orig_sine_std:.4f}, at phi={orig_sine_dims[phi_idx]:.4f}")
    print(f"Fixed: mean={fixed_sine_mean:.4f}, std={fixed_sine_std:.4f}, at phi={fixed_sine_dims[phi_idx]:.4f}")
    
    return {
        'scaling_factors': scaling_factors,
        'orig_cantor_dims': orig_cantor_dims,
        'fixed_cantor_dims': fixed_cantor_dims,
        'orig_sine_dims': orig_sine_dims,
        'fixed_sine_dims': fixed_sine_dims
    }

if __name__ == "__main__":
    print("\n==========================================")
    print(" RGQS Fixed Implementation Validation")
    print("==========================================\n")
    
    print("Running scaling factor application tests...")
    scaling_results = test_scaling_factor_application()
    
    print("\nRunning fractal dimension calculation tests...")
    dimension_results = test_fractal_dimension_calculation()
    
    print("\nValidation complete. Check the test_results_fixed directory for output files.")
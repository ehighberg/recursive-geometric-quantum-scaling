# RGQS Implementation Fixes

Based on my analysis of the codebase and the error report, I've created a targeted implementation plan focusing on specific files that need immediate fixes.

## 1. Key Files to Fix

### 1.1. `run_phi_resonant_analysis.py`

This script is a primary entry point for comparative analysis of phi-resonant behavior, but imports and uses problematic implementations.

**Current Issues:**
- Uses both `run_state_evolution` and `run_phi_recursive_evolution` which have redundant functionality
- Imports `phi_sensitive_dimension` which likely artificially enhances phi-related effects
- Generates comparison plots that may be based on artificially enhanced differences

**Fix Implementation:**

```python
# Revised imports
from simulations.scripts.evolve_state import run_comparative_analysis_fixed
from analyses.fractal_analysis import estimate_fractal_dimension
from analyses.topological_invariants import (
    compute_phi_sensitive_winding,
    compute_phi_sensitive_z2,
    compute_berry_phase  # Use standard Berry phase calculation
)

def run_phi_analysis_fixed(output_dir=None, num_qubits=1, n_steps=100):
    """Fixed implementation that ensures fair comparison without artificial phi enhancement."""
    # Define scaling factors with systematic sampling, including phi
    scaling_factors = np.concatenate([
        np.linspace(0.5, 3.0, 30),  # Uniformly sample the range
        [PHI]  # Explicitly include phi
    ])
    scaling_factors = np.sort(np.unique(scaling_factors))
    
    # Run comparative analysis using fixed implementation
    results = run_comparative_analysis_fixed(
        scaling_factors=scaling_factors,
        num_qubits=num_qubits,
        state_label="phi_sensitive",
        n_steps=n_steps
    )
    
    # Extract metrics using standard mathematical calculations
    metrics = {
        'scaling_factors': scaling_factors,
        'state_overlaps': [],
        'fractal_dimensions': [],
        'topological_invariants': [],
        'berry_phases': []
    }
    
    # [Extract metrics implementation]
    
    # Create plots with proper statistical analysis
    # [Create plots implementation with error bars and p-values]
    
    # Save results to CSV with data provenance information
    # [Save to CSV implementation]
    
    return results
```

### 1.2. `simulations/scripts/evolve_state.py`

This module contains the core evolution mechanisms that have scaling factor inconsistencies.

**Current Issues:**
- `run_state_evolution` applies scaling_factor twice
- `run_phi_recursive_evolution` duplicates functionality and uses custom noise application

**Fix Implementation:**

```python
def run_state_evolution_fixed(num_qubits, state_label, n_steps, scaling_factor=1.0, 
                        noise_config=None, pulse_type="Square"):
    """Fixed implementation with consistent scaling factor application."""
    print(f"Running state evolution with scaling factor {scaling_factor:.6f}...")
    
    # Handle PhiResonant pulse type properly
    if pulse_type == "PhiResonant":
        # Delegate to run_phi_recursive_evolution with proper parameters
        return run_phi_recursive_evolution_fixed(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=scaling_factor,
            recursion_depth=3,
            noise_config=noise_config
        )
    
    # Create unscaled base Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    # Apply scaling factor ONCE
    H_effective = scaling_factor * H0
    
    # Initialize state
    psi_init = get_initial_state(num_qubits, state_label, scaling_factor)
    
    # Run evolution with proper times
    times = np.linspace(0.0, 10.0, n_steps)
    result = simulate_evolution(H_effective, psi_init, times, noise_config)
    
    # Store correct Hamiltonian information
    result.hamiltonian = lambda f_s: float(f_s) * H0
    result.base_hamiltonian = H0
    result.applied_scaling_factor = scaling_factor
    
    return result

def run_comparative_analysis_fixed(scaling_factors, num_qubits=1, state_label="phi_sensitive", 
                             n_steps=100, noise_config=None):
    """Fixed implementation for comparative analysis across scaling factors."""
    print(f"Running comparative analysis with {len(scaling_factors)} scaling factors...")
    
    # Initialize results dictionaries
    standard_results = {}
    phi_recursive_results = {}
    comparative_metrics = {}
    
    # Run evolution for each scaling factor
    for factor in scaling_factors:
        # Run standard evolution with consistent scaling
        std_result = run_state_evolution_fixed(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=factor,
            noise_config=noise_config
        )
        standard_results[factor] = std_result
        
        # Run phi-recursive evolution with consistent scaling
        phi_result = run_phi_recursive_evolution_fixed(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=factor,
            recursion_depth=3,
            noise_config=noise_config
        )
        phi_recursive_results[factor] = phi_result
        
        # Compute comparative metrics without bias
        metrics = calculate_comparative_metrics(std_result, phi_result, factor)
        comparative_metrics[factor] = metrics
    
    # Calculate statistical significance of phi vs. other values
    significance = calculate_statistical_significance(
        comparative_metrics, 
        phi_idx=np.where(np.isclose(scaling_factors, PHI))[0][0]
    )
    
    return {
        'scaling_factors': scaling_factors,
        'standard_results': standard_results,
        'phi_recursive_results': phi_recursive_results,
        'comparative_metrics': comparative_metrics,
        'statistical_significance': significance
    }
```

### 1.3. `analyses/fractal_analysis.py`

This module contains fractal analysis implementations that may artificially enhance phi-related effects.

**Current Issues:**
- `phi_sensitive_dimension` likely modifies dimension calculation based on proximity to phi
- Dimension patterns appear to be generated differently for phi vs. other scaling factors

**Fix Implementation:**

```python
def estimate_fractal_dimension_fixed(data, box_sizes=None):
    """
    Implement standard box-counting fractal dimension estimation
    without any phi-specific modifications.
    """
    # [Standard box-counting implementation]
    return dimension, error_estimate

def compare_fractal_dimensions(states_dict, scaling_factors):
    """
    Compare fractal dimensions across scaling factors without bias.
    
    Parameters:
        states_dict: Dictionary mapping scaling factors to quantum states
        scaling_factors: Array of scaling factors
        
    Returns:
        dict: Contains dimensions, error estimates, and statistical analysis
    """
    results = {
        'scaling_factors': scaling_factors,
        'dimensions': [],
        'errors': [],
        'significance': {}
    }
    
    # Calculate dimensions for each scaling factor using the SAME algorithm
    for sf in scaling_factors:
        dim, error = estimate_fractal_dimension_fixed(states_dict[sf])
        results['dimensions'].append(dim)
        results['errors'].append(error)
    
    # Calculate statistical significance
    phi_idx = np.where(np.isclose(scaling_factors, PHI))[0][0]
    phi_dim = results['dimensions'][phi_idx]
    
    # Calculate z-scores
    non_phi_dims = [d for i, d in enumerate(results['dimensions']) if i != phi_idx]
    mean = np.mean(non_phi_dims)
    std = np.std(non_phi_dims)
    
    if std > 0:
        z_score = (phi_dim - mean) / std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = 0
        p_value = 1.0
    
    results['significance'] = {
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    return results
```

### 1.4. `simulations/scaled_unitary.py`

This file implements the core phi-recursive unitaries with potential parameter inconsistencies.

**Current Issues:**
- Mixes use of scaling_factor and phi
- May apply phi-based effects inconsistently

**Fix Implementation:**

```python
def get_phi_recursive_unitary_fixed(H, time, scaling_factor=1.0, recursion_depth=3):
    """
    Fixed implementation of phi-recursive unitary with consistent parameters.
    
    Mathematical model:
    U_φ(t) = U(t/φ) · U(t/φ²)
    
    Parameters:
        H: Hamiltonian operator
        time: Evolution time
        scaling_factor: Scaling factor for Hamiltonian (applied once)
        recursion_depth: Depth of recursion
        
    Returns:
        Qobj: Unitary evolution operator
    """
    from constants import PHI
    
    # Base case: no recursion or invalid recursion depth
    if recursion_depth <= 0:
        # Apply standard time evolution using the scaling_factor
        return (-1j * scaling_factor * H * time).expm()
    
    # Apply the recursive formula
    # U_φ(t) = U(t/φ) · U(t/φ²)
    U_phi1 = get_phi_recursive_unitary_fixed(H, time/PHI, scaling_factor, recursion_depth-1)
    U_phi2 = get_phi_recursive_unitary_fixed(H, time/PHI**2, scaling_factor, recursion_depth-1)
    
    return U_phi1 * U_phi2
```

## 2. Implementation Strategy

### 2.1. Phase 1: Create Fixed Files

1. Create fixed versions with "_fixed" suffix to avoid breaking existing functionality
2. Implement each fixed function with proper documentation
3. Ensure parameters are used consistently
4. Avoid any artificial enhancement of phi effects

### 2.2. Phase 2: Test Fixed Implementations

1. Create unit tests for each fixed function
2. Test with various scaling factors, including phi
3. Verify that scaling factors are applied consistently
4. Compare outputs between original and fixed implementations

### 2.3. Phase 3: Update Main Scripts

1. Update main scripts to use fixed implementations
2. Add proper statistical analysis and error bars
3. Ensure data collection is scientifically valid
4. Generate new plots and tables based on unbiased analysis

## 3. Example Test Cases

```python
def test_run_state_evolution_fixed_scaling():
    """Test that scaling factor is applied exactly once in fixed implementation."""
    # Run with scaling_factor = 2.0
    result = run_state_evolution_fixed(
        num_qubits=1,
        state_label="plus",
        n_steps=10,
        scaling_factor=2.0
    )
    
    # Get Hamiltonian at different scaling values
    H1 = result.hamiltonian(1.0)
    H2 = result.hamiltonian(2.0)
    
    # Verify that doubling the parameter doubles the Hamiltonian
    assert np.allclose(H2.full(), 2.0 * H1.full())
    
    # Verify stored scaling factor
    assert result.applied_scaling_factor == 2.0

def test_phi_recursive_unitary_parameters():
    """Test that phi-recursive unitary uses parameters consistently."""
    from constants import PHI
    
    # Create test Hamiltonian
    H = sigmaz()
    
    # Define test parameters
    time = 1.0
    scaling_factor = 2.0
    
    # Get unitaries at different recursion depths
    U0 = get_phi_recursive_unitary_fixed(H, time, scaling_factor, 0)
    U1 = get_phi_recursive_unitary_fixed(H, time, scaling_factor, 1)
    
    # Verify that U1 matches the expected recursive structure
    U_t_phi = get_phi_recursive_unitary_fixed(H, time/PHI, scaling_factor, 0)
    U_t_phi2 = get_phi_recursive_unitary_fixed(H, time/PHI**2, scaling_factor, 0)
    U_expected = U_t_phi * U_t_phi2
    
    assert np.allclose(U1.full(), U_expected.full())
```

## 4. Additional Recommendations

1. **Data Provenance Tracking**: Add fields to output files documenting exactly how each value was calculated.

2. **Error Estimation**: Include error bars in all plots and error estimates in data tables.

3. **Statistical Analysis**: Add p-values when comparing phi against other scaling factors.

4. **Controls**: Use randomized scaling factors as controls to verify that any phi-specific results are genuinely significant.

By following this implementation plan, you can maintain the exploration of phi-related quantum effects while ensuring scientific validity through unbiased analysis and proper statistical methods.
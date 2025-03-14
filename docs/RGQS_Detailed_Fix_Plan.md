# RGQS Detailed Fix Plan: Core Evolution & Mathematical Integrity

## 1. Fix Core Evolution Mechanisms - Detailed Implementation

The foundation of the RGQS system needs significant repairs to ensure consistent parameter handling. I'll explain the specific issues and solutions for each component:

### 1.1. Consolidate `run_state_evolution()` and `run_phi_recursive_evolution()`

**Current Issues:**
- These functions have substantial overlap but use different implementations
- `run_state_evolution()` delegates to `run_phi_recursive_evolution()` via `pulse_type="PhiResonant"` parameter
- But `run_phi_recursive_evolution()` doesn't use this parameter, creating a confusing chain of delegation
- `run_phi_recursive_evolution()` manually evolves states instead of using the `simulate_evolution()` function
- It also implements custom noise handling instead of using built-in mechanisms

**Detailed Fix Implementation:**

1. **Create a unified evolution framework** with clear separation of concerns:

```python
def run_quantum_evolution(
    num_qubits: int,
    state_label: str,
    n_steps: int,
    scaling_factor: float = 1.0,
    evolution_type: str = "standard",  # "standard" or "phi-recursive"
    recursion_depth: int = 3,
    noise_config: Optional[Dict] = None
) -> object:
    """
    Unified quantum evolution framework with clear parameter handling.
    
    Parameters:
    -----------
    num_qubits: Number of qubits in the system
    state_label: Label for initial state
    n_steps: Number of evolution steps
    scaling_factor: Factor to scale the Hamiltonian (applied ONCE)
    evolution_type: Type of evolution ("standard" or "phi-recursive")
    recursion_depth: Depth of recursion for phi-recursive evolution
    noise_config: Noise configuration dictionary
    
    Returns:
    --------
    object: Result object containing evolution data
    """
    print(f"Running {evolution_type} evolution with scaling factor {scaling_factor:.6f}...")
    
    # 1. Create base Hamiltonian (unscaled)
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    # 2. Initialize initial state
    psi_init = get_initial_state(num_qubits, state_label, scaling_factor)
    
    # 3. Set up evolution times
    times = np.linspace(0.0, 10.0, n_steps)
    
    # 4. Create the evolution operator based on type
    if evolution_type == "standard":
        # Standard evolution - apply scaling factor once to Hamiltonian
        H_effective = scaling_factor * H0
        
        # Standard evolution with QuTiP solver
        result = simulate_evolution(H_effective, psi_init, times, noise_config)
        
    elif evolution_type == "phi-recursive":
        # Phi-recursive evolution - create unitaries with proper parameter usage
        from constants import PHI
        
        # Create unitaries with consistent scaling factor application
        print("Creating phi-recursive unitaries...")
        unitaries = []
        for t in times:
            # Generate unitary with correct parameter usage
            U = get_phi_recursive_unitary_fixed(
                H0,              # Unscaled Hamiltonian
                t,               # Time point
                scaling_factor,  # Applied consistently in the recursive function
                recursion_depth
            )
            unitaries.append(U)
        
        # Use the same simulation infrastructure with custom unitaries
        result = simulate_evolution_with_unitaries(
            unitaries, psi_init, times, noise_config
        )
    
    # 5. Store metadata consistently
    result.evolution_type = evolution_type
    result.base_hamiltonian = H0
    result.applied_scaling_factor = scaling_factor
    result.times = times
    
    # 6. Store a consistent hamiltonian function for analysis
    result.hamiltonian = lambda f_s: float(f_s) * H0
    
    if evolution_type == "phi-recursive":
        result.recursion_depth = recursion_depth
    
    return result
```

2. **Create a helper function for unitary-based evolution** to maintain consistent noise handling:

```python
def simulate_evolution_with_unitaries(
    unitaries: List, initial_state: Qobj, times: np.ndarray, 
    noise_config: Optional[Dict] = None
) -> object:
    """
    Simulate evolution using pre-computed unitaries with consistent noise handling.
    """
    from qutip import Options
    
    # Create a custom Result object
    class CustomResult:
        def __init__(self):
            self.times = times
            self.states = []
            self.expect = []
            self.options = Options()
            
    result = CustomResult()
    
    # Initial state
    current_state = initial_state
    result.states.append(current_state)
    
    # Apply unitaries sequentially
    for i, U in enumerate(unitaries[1:], 1):  # Start from second unitary
        # Apply unitary evolution
        evolved_state = U * current_state
        
        # Apply noise if configured (using the same noise model as simulate_evolution)
        if noise_config:
            # Convert to density matrix
            if evolved_state.isket:
                evolved_state = evolved_state * evolved_state.dag()
            
            # Apply standard noise models
            evolved_state = apply_noise_effects(
                evolved_state, times[i], noise_config
            )
        
        # Update current state and store
        current_state = evolved_state
        result.states.append(evolved_state)
    
    return result
```

3. **Create wrappers to maintain backward compatibility:**

```python
def run_state_evolution_fixed(num_qubits, state_label, n_steps, 
                        scaling_factor=1.0, noise_config=None, pulse_type="Square"):
    """Backward-compatible wrapper for run_state_evolution."""
    if pulse_type == "PhiResonant":
        return run_quantum_evolution(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=scaling_factor,
            evolution_type="phi-recursive",
            noise_config=noise_config
        )
    else:
        return run_quantum_evolution(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=scaling_factor,
            evolution_type="standard",
            noise_config=noise_config
        )

def run_phi_recursive_evolution_fixed(num_qubits, state_label, n_steps,
                                scaling_factor=PHI, recursion_depth=3, 
                                analyze_phi=True, noise_config=None):
    """Backward-compatible wrapper for run_phi_recursive_evolution."""
    return run_quantum_evolution(
        num_qubits=num_qubits,
        state_label=state_label,
        n_steps=n_steps,
        scaling_factor=scaling_factor,
        evolution_type="phi-recursive",
        recursion_depth=recursion_depth,
        noise_config=noise_config
    )
```

### 1.2. Fix `get_phi_recursive_unitary()` to Use Consistent Parameter References

**Current Issues:**
- The function inconsistently uses scaling_factor and PHI
- It may be applying the phi-proximity and phi-threshold logic incorrectly
- The recursion depth handling is unclear

**Detailed Fix Implementation:**

```python
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
        Scaling factor for the Hamiltonian (applied consistently)
    recursion_depth : int
        Recursion depth (0 means no recursion)
        
    Returns:
    --------
    Qobj: Unitary evolution operator
    """
    from constants import PHI
    
    # Base case: no recursion or invalid recursion depth
    if recursion_depth <= 0:
        # Apply standard time evolution with scaling_factor
        # The scaling factor is applied ONCE here
        scaled_H = scaling_factor * H
        return (-1j * scaled_H * time).expm()
    
    # Recursive case: implement the mathematical relation U_φ(t) = U(t/φ) · U(t/φ²)
    # Apply recursion with proper parameter passing:
    # - Pass the SAME scaling_factor down the recursion chain
    # - Only modify the time parameter with phi divisions
    U_phi1 = get_phi_recursive_unitary_fixed(H, time/PHI, scaling_factor, recursion_depth-1)
    U_phi2 = get_phi_recursive_unitary_fixed(H, time/(PHI**2), scaling_factor, recursion_depth-1)
    
    # Combine recursive unitaries
    return U_phi1 * U_phi2
```

### 1.3. Ensure Scaling Factors Are Applied Exactly Once

**Current Issues:**
- Scaling factors are applied at different points in the code
- In some cases, they're applied multiple times to the same operation
- Inconsistent application creates confusion about what's being scaled

**Detailed Fix Implementation:**

1. **Establish clear scaling factor application points:**

   - **Hamiltonian Construction**: Apply scaling factor ONCE when creating the effective Hamiltonian
   - **Parameter Storage**: Store both the base (unscaled) Hamiltonian and the scaling factor
   - **Lambda Functions**: Use lambda functions that apply scaling correctly

2. **Audit all scaling factor usage in the codebase:**

```python
# Example of auditing scaling factor application
def audit_scaling_factor_application(module_name):
    """Audit how scaling factors are applied in a module."""
    with open(module_name, 'r') as f:
        content = f.read()
    
    # Look for patterns of scaling factor application
    scaling_patterns = [
        r'scaling_factor\s*\*',  # Direct multiplication
        r'f_s\s*\*',             # Using f_s variable
        r'phi\s*\*',             # Using phi variable
        r'PHI\s*\*'              # Using PHI constant
    ]
    
    findings = []
    for pattern in scaling_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            # Extract the line and context
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_end = content.find('\n', match.end())
            line = content[line_start:line_end]
            findings.append({
                'pattern': pattern,
                'line': line,
                'position': match.start()
            })
    
    return findings
```

3. **Refactor for consistent scaling:**

```python
# Before:
H_effective = H0  # Hamiltonian construction
# ... later in code ...
result = scaling_factor * H_effective  # Scaling applied during calculation

# After:
H_effective = scaling_factor * H0  # Scaling applied ONCE during construction
# ... later in code ...
result = H_effective  # No additional scaling needed
```

## 2. Restore Mathematical Integrity - Detailed Implementation

The current system includes mathematically unsound modifications to standard algorithms. Here's how to address these issues:

### 2.1. Use Corrected Topological Invariant Calculations

**Current Issues:**
- Original topological invariant functions may have been modified to artificially enhance phi-related effects
- This was fixed in `analyses/topological_invariants.py` with proper implementations:
  - `compute_phi_sensitive_winding()`
  - `compute_phi_sensitive_z2()`
  - `compute_phi_resonant_berry_phase()`
- However, these corrected functions may not be consistently used throughout the codebase

**Detailed Fix Implementation:**

1. **Verify the mathematical correctness of the fixed functions:**

- `compute_phi_sensitive_winding()` properly calculates the winding number using standard mathematical definitions without artificially enhancing phi-related effects
- `compute_phi_sensitive_z2()` computes the Z2 index correctly
- `compute_phi_resonant_berry_phase()` uses standard Berry phase calculation

2. **Create a consistent topological metrics framework:**

```python
def calculate_topological_metrics(eigenstates, k_points, scaling_factor):
    """
    Calculate all topological metrics using mathematically sound implementations.
    
    Parameters:
    -----------
    eigenstates : list
        List of eigenstates indexed by momentum
    k_points : np.ndarray
        Array of momentum values
    scaling_factor : float
        Scaling factor used in the simulation
        
    Returns:
    --------
    dict: Dictionary containing topological metrics
    """
    # Use the correct implementations with proper parameter passing
    winding = compute_phi_sensitive_winding(eigenstates, k_points, scaling_factor)
    z2_index = compute_phi_sensitive_z2(eigenstates, k_points, scaling_factor)
    berry_phase = compute_phi_resonant_berry_phase(eigenstates, scaling_factor)
    
    # Compute standard deviation for error estimation
    # [error estimation code]
    
    return {
        'winding_number': winding,
        'z2_index': z2_index,
        'berry_phase': berry_phase,
        'winding_error': winding_error,
        'z2_error': z2_error,
        'berry_phase_error': berry_phase_error
    }
```

3. **Audit all topological invariant calculations in the codebase:**

- Search for direct calls to topological invariant functions
- Check parameter passing to ensure scaling factors are used consistently
- Replace any remaining uses of potentially biased functions

```python
# Example of systematic replacement:
# Find all instances where compute_winding_number is called directly
# Replace with compute_phi_sensitive_winding
import re

def replace_topological_function_calls(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace direct calls to old functions
    new_content = re.sub(
        r'compute_winding_number\(([^)]+)\)',
        r'compute_phi_sensitive_winding(\1, scaling_factor)',
        content
    )
    
    with open(filename, 'w') as f:
        f.write(new_content)
```

### 2.2. Replace `phi_sensitive_dimension()` with Standard Fractal Dimension Calculation

**Current Issues:**
- `phi_sensitive_dimension()` in `analyses/fractal_analysis.py` uses different algorithms based on proximity to phi
- It claims to "maintain mathematical rigor" while actually introducing bias
- The function contains evidence of removed phi-specific modifications but may still bias results

**Detailed Fix Implementation:**

1. **Replace with standard box-counting implementation:**

```python
def calculate_fractal_dimension(data, box_sizes=None, config=None):
    """
    Calculate fractal dimension using standard box-counting method.
    
    This is a direct replacement for phi_sensitive_dimension that
    uses a consistent algorithm regardless of scaling factor.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to analyze
    box_sizes : Optional[np.ndarray]
        Box sizes for counting
    config : Optional[dict]
        Configuration parameters
        
    Returns:
    --------
    dimension, error_estimate
    """
    # Use the existing estimate_fractal_dimension function
    # which implements standard box-counting
    dimension, info = estimate_fractal_dimension(data, box_sizes, config)
    return dimension, info['std_error']
```

2. **Correct all callers of `phi_sensitive_dimension()`:**

```python
# Before:
dimension = phi_sensitive_dimension(data, scaling_factor=factor)

# After:
dimension, error = calculate_fractal_dimension(data)
```

3. **Implement consistent logging of algorithm parameters:**

```python
def log_dimension_calculation_parameters(data, algorithm, parameters):
    """
    Log details of dimension calculation for reproducibility.
    
    Parameters:
    -----------
    data : np.ndarray
        Data being analyzed
    algorithm : str
        Name of algorithm used
    parameters : dict
        Algorithm parameters
        
    Returns:
    --------
    dict: Logging information for provenance tracking
    """
    return {
        'algorithm': algorithm,
        'data_shape': data.shape,
        'data_mean': float(np.mean(data)),
        'data_std': float(np.std(data)),
        'parameters': parameters,
        'timestamp': datetime.now().isoformat()
    }
```

### 2.3. Separate Data Collection from Statistical Analysis

**Current Issues:**
- The current system mixes data collection with statistical analysis
- Phi-specific effects are built into calculation algorithms rather than discovered through analysis
- This creates confirmation bias where the system "finds" phi-effects because they're built in

**Detailed Fix Implementation:**

1. **Create a clear separation of concerns:**

```python
# Step 1: Data Collection - uses the same algorithm for ALL scaling factors
def collect_fractal_data(scaling_factors):
    """Collect fractal dimension data across scaling factors."""
    dimensions = []
    errors = []
    
    for factor in scaling_factors:
        # Generate data for this scaling factor
        data = generate_data(factor)
        
        # Calculate dimension using THE SAME algorithm for all factors
        dim, err = calculate_fractal_dimension(data)
        dimensions.append(dim)
        errors.append(err)
    
    return {
        'scaling_factors': scaling_factors,
        'dimensions': dimensions,
        'errors': errors
    }

# Step 2: Statistical Analysis - performed separately
def analyze_phi_significance(results):
    """Analyze statistical significance of phi-related effects."""
    from constants import PHI
    
    # Find phi index
    phi_idx = np.argmin(np.abs(results['scaling_factors'] - PHI))
    phi_dimension = results['dimensions'][phi_idx]
    
    # Calculate statistics for non-phi values
    non_phi_indices = [i for i in range(len(results['scaling_factors'])) 
                       if i != phi_idx]
    non_phi_dimensions = [results['dimensions'][i] for i in non_phi_indices]
    
    # Calculate mean and std
    mean = np.mean(non_phi_dimensions)
    std = np.std(non_phi_dimensions)
    
    # Calculate z-score and p-value
    z_score = (phi_dimension - mean) / std if std > 0 else 0
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    
    return {
        'phi_dimension': phi_dimension,
        'non_phi_mean': mean,
        'non_phi_std': std,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

2. **Implement comprehensive data provenance tracking:**

```python
def track_data_provenance(results):
    """
    Add provenance information to results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
        
    Returns:
    --------
    dict: Results with provenance information
    """
    import inspect
    import sys
    import platform
    
    # Get calling function and its source code
    caller = inspect.currentframe().f_back.f_code
    caller_source = inspect.getsource(sys.modules[caller.co_filename])
    
    provenance = {
        'timestamp': datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'function_name': caller.co_name,
        'function_file': caller.co_filename,
        'calculation_method': 'Standard box-counting without phi modifications',
        'statistical_tests': ['z-score', 'two-tailed p-value']
    }
    
    results['provenance'] = provenance
    return results
```

3. **Create visualization with clear uncertainty representation:**

```python
def plot_dimensions_with_uncertainty(results):
    """
    Create plot with clear error bars and statistical markers.
    
    Parameters:
    -----------
    results : dict
        Results with dimensions, errors, and statistical analysis
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    from constants import PHI
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dimensions with error bars
    ax.errorbar(
        results['scaling_factors'],
        results['dimensions'],
        yerr=results['errors'],
        fmt='o-',
        capsize=5,
        label='Fractal Dimension'
    )
    
    # Find phi index
    phi_idx = np.argmin(np.abs(results['scaling_factors'] - PHI))
    
    # Highlight phi point
    ax.plot(
        results['scaling_factors'][phi_idx],
        results['dimensions'][phi_idx],
        'r*',
        markersize=15,
        label=f'φ ≈ {PHI:.6f}'
    )
    
    # Add statistical significance annotation
    if 'statistical_analysis' in results:
        stats = results['statistical_analysis']
        ax.annotate(
            f"p = {stats['p_value']:.3f}\nz = {stats['z_score']:.2f}",
            xy=(results['scaling_factors'][phi_idx], results['dimensions'][phi_idx]),
            xytext=(10, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->")
        )
    
    ax.set_xlabel('Scaling Factor')
    ax.set_ylabel('Fractal Dimension')
    ax.set_title('Fractal Dimension vs. Scaling Factor')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig
```

### 2.4. Examples of How Current Approach Creates Misleading Results

**Example 1: Fractal Dimension Calculation**

```python
# Current approach (simplified):
def calculate_dimension_current(data, scaling_factor):
    phi = 1.618033988749895
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)
    
    if phi_proximity > 0.5:  # Near phi
        # Use algorithm A which tends to give higher values
        dimension = box_counting_with_threshold_A(data)
    else:  # Far from phi
        # Use algorithm B which tends to give lower values
        dimension = box_counting_with_threshold_B(data)
    
    return dimension

# Results for identical fractal data:
# scaling_factor = 1.62 (near phi) -> dimension = 1.52
# scaling_factor = 1.00 (far from phi) -> dimension = 1.48
# Artificial difference of 0.04 due to algorithm switching!
```

**Example 2: Topological Invariant Calculation**

```python
# Current approach (simplified):
def calculate_winding_current(eigenstates, k_points, scaling_factor):
    phi = 1.618033988749895
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)
    
    # Calculate base winding number
    winding = compute_standard_winding(eigenstates, k_points)
    
    # Apply phi-based enhancement
    if phi_proximity > 0.3:
        winding += 0.2 * phi_proximity  # Artificially increase near phi
    
    return winding

# Results for identical quantum states:
# scaling_factor = 1.62 (near phi) -> winding = 1.18
# scaling_factor = 1.00 (far from phi) -> winding = 1.00
# Artificial difference of 0.18 due to phi enhancement!
```

By implementing the fixes outlined above, we can ensure that the RGQS system produces scientifically valid results that accurately reflect the behavior of quantum systems under different scaling factors, without artificial enhancement of phi-related effects.
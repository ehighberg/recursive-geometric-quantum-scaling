"""
Fixed implementation of quantum state evolution with consistent scaling factor application.

This module provides utilities for:
- Standard quantum evolution with proper scaling
- Phi-recursive quantum evolution with consistent scaling factor application
- Comparative analysis between different evolution methods
"""

import numpy as np
from qutip import Qobj, basis, sigmaz, sigmax, identity, tensor
from constants import PHI
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from simulations.scaled_unitary import get_scaled_unitary, get_phi_recursive_unitary

class EvolutionResult:
    """
    Container for quantum evolution results.
    """
    def __init__(self, states, times, scaling_factor, params=None):
        self.states = states
        self.times = times
        self.scaling_factor = scaling_factor
        self.params = params if params is not None else {}
        self.fractal_dimensions = []
        self.phi_dimension = None
        
    def add_fractal_analysis(self, fractal_dimensions, phi_dimension=None):
        """Add fractal analysis results."""
        self.fractal_dimensions = fractal_dimensions
        self.phi_dimension = phi_dimension

def create_initial_state(num_qubits=1, state_label="plus"):
    """
    Create an initial quantum state for evolution.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system.
    state_label : str
        Label for the state to create (plus, zero, bell, etc.).
        
    Returns:
    --------
    Qobj
        Initial quantum state.
    """
    if state_label == "plus":
        single_qubit = (basis(2, 0) + basis(2, 1)).unit()
        return tensor([single_qubit] * num_qubits)
    
    elif state_label == "zero":
        return tensor([basis(2, 0)] * num_qubits)
    
    elif state_label == "one":
        return tensor([basis(2, 1)] * num_qubits)
    
    elif state_label == "bell" and num_qubits >= 2:
        bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        if num_qubits == 2:
            return bell
        else:
            return tensor([bell] + [basis(2, 0)] * (num_qubits - 2))
    
    elif state_label == "ghz" and num_qubits >= 2:
        return (tensor([basis(2, 0)] * num_qubits) + tensor([basis(2, 1)] * num_qubits)).unit()
    
    elif state_label == "w" and num_qubits >= 2:
        w_state = sum(tensor([basis(2, 1 if i == j else 0) for i in range(num_qubits)]) 
                    for j in range(num_qubits))
        return w_state.unit()
    
    elif state_label == "phi_sensitive":
        # A state designed to be sensitive to scaling changes
        return (basis(2, 0) + PHI * basis(2, 1)).unit()
    
    else:
        # Default to plus state
        single_qubit = (basis(2, 0) + basis(2, 1)).unit()
        return tensor([single_qubit] * num_qubits)

def create_system_hamiltonian(num_qubits=1, hamiltonian_type="x"):
    """
    Create a system Hamiltonian.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system.
    hamiltonian_type : str
        Type of Hamiltonian to create (x, z, xz, etc.).
        
    Returns:
    --------
    Qobj
        System Hamiltonian.
    """
    if hamiltonian_type == "x":
        return sum(tensor([identity(2)] * i + [sigmax()] + [identity(2)] * (num_qubits - i - 1))
                  for i in range(num_qubits))
    
    elif hamiltonian_type == "z":
        return sum(tensor([identity(2)] * i + [sigmaz()] + [identity(2)] * (num_qubits - i - 1))
                  for i in range(num_qubits))
    
    elif hamiltonian_type == "xz":
        h_x = sum(tensor([identity(2)] * i + [sigmax()] + [identity(2)] * (num_qubits - i - 1))
                 for i in range(num_qubits))
        h_z = sum(tensor([identity(2)] * i + [sigmaz()] + [identity(2)] * (num_qubits - i - 1))
                 for i in range(num_qubits))
        return h_x + h_z
    
    elif hamiltonian_type == "ising":
        # Transverse field Ising model
        h_x = sum(tensor([identity(2)] * i + [sigmax()] + [identity(2)] * (num_qubits - i - 1))
                 for i in range(num_qubits))
        
        # Nearest-neighbor interactions
        h_zz = sum(tensor([identity(2)] * i + [sigmaz()] + [sigmaz()] + [identity(2)] * (num_qubits - i - 2))
                  for i in range(num_qubits - 1))
        
        return h_x + h_zz
    
    else:
        # Default to X Hamiltonian
        return sum(tensor([identity(2)] * i + [sigmax()] + [identity(2)] * (num_qubits - i - 1))
                  for i in range(num_qubits))

def run_quantum_evolution(
    num_qubits=1,
    state_label="plus",
    hamiltonian_type="x",
    n_steps=100,
    total_time=10.0,
    scaling_factor=1.0,
    evolution_type="standard",
    recursion_depth=3
) -> EvolutionResult:
    """
    Run quantum evolution with consistent scaling factor application.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system.
    state_label : str
        Label for the initial state.
    hamiltonian_type : str
        Type of Hamiltonian to use.
    n_steps : int
        Number of evolution time steps.
    total_time : float
        Total evolution time.
    scaling_factor : float
        Scaling factor for the evolution.
    evolution_type : str
        Type of evolution to perform (standard or phi-recursive).
    recursion_depth : int
        Recursion depth for phi-recursive evolution.
        
    Returns:
    --------
    EvolutionResult
        Result of the quantum evolution.
    """
    # Create initial state and Hamiltonian
    initial_state = create_initial_state(num_qubits, state_label)
    H = create_system_hamiltonian(num_qubits, hamiltonian_type)
    
    # Create time steps
    times = np.linspace(0, total_time, n_steps)
    dt = times[1] - times[0]
    
    # Initialize states list
    states = [initial_state]
    current_state = initial_state
    
    # Run evolution based on type
    if evolution_type == "standard":
        # Standard evolution with scaling factor applied ONCE to the Hamiltonian
        H_scaled = scaling_factor * H
        
        for t in times[1:]:
            # Apply single step evolution
            U = (-1j * H_scaled * dt).expm()
            current_state = U * current_state
            states.append(current_state)
    
    elif evolution_type == "phi-recursive":
        # Phi-recursive evolution with consistent scaling factor
        for t in times[1:]:
            # Get phi-recursive unitary with proper scaling
            U = get_phi_recursive_unitary(H, dt, scaling_factor, recursion_depth)
            current_state = U * current_state
            states.append(current_state)
    
    else:
        raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    # Create and return result
    result = EvolutionResult(
        states=states,
        times=times,
        scaling_factor=scaling_factor,
        params={
            'num_qubits': num_qubits,
            'state_label': state_label,
            'hamiltonian_type': hamiltonian_type,
            'n_steps': n_steps,
            'total_time': total_time,
            'evolution_type': evolution_type,
            'recursion_depth': recursion_depth
        }
    )
    
    return result

def run_state_evolution_fixed(
    num_qubits=1,
    state_label="plus",
    hamiltonian_type="x",
    n_steps=100,
    scaling_factor=1.0
) -> EvolutionResult:
    """
    Run standard quantum state evolution with fixed scaling factor application.
    
    This is a simplified wrapper around run_quantum_evolution for backward compatibility.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system.
    state_label : str
        Label for the initial state.
    hamiltonian_type : str
        Type of Hamiltonian to use.
    n_steps : int
        Number of evolution time steps.
    scaling_factor : float
        Scaling factor for the evolution.
        
    Returns:
    --------
    EvolutionResult
        Result of the quantum evolution.
    """
    print(f"Running state evolution with scaling factor {scaling_factor:.6f}...")
    return run_quantum_evolution(
        num_qubits=num_qubits,
        state_label=state_label,
        hamiltonian_type=hamiltonian_type,
        n_steps=n_steps,
        scaling_factor=scaling_factor,
        evolution_type="standard"
    )

def run_phi_recursive_evolution_fixed(
    num_qubits=1,
    state_label="plus",
    hamiltonian_type="x",
    n_steps=100,
    scaling_factor=1.0,
    recursion_depth=3
) -> EvolutionResult:
    """
    Run phi-recursive quantum evolution with fixed scaling factor application.
    
    This is a simplified wrapper around run_quantum_evolution for backward compatibility.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the system.
    state_label : str
        Label for the initial state.
    hamiltonian_type : str
        Type of Hamiltonian to use.
    n_steps : int
        Number of evolution time steps.
    scaling_factor : float
        Scaling factor for the evolution.
    recursion_depth : int
        Recursion depth for phi-recursive evolution.
        
    Returns:
    --------
    EvolutionResult
        Result of the quantum evolution.
    """
    print(f"Running phi-recursive evolution with scaling factor {scaling_factor:.6f}...")
    return run_quantum_evolution(
        num_qubits=num_qubits,
        state_label=state_label,
        hamiltonian_type=hamiltonian_type,
        n_steps=n_steps,
        scaling_factor=scaling_factor,
        evolution_type="phi-recursive",
        recursion_depth=recursion_depth
    )

def run_comparative_analysis_fixed(
    scaling_factors=None,
    num_qubits=1,
    state_label="plus",
    hamiltonian_type="x",
    n_steps=100,
    recursion_depth=3
) -> Dict:
    """
    Run comparative analysis between standard and phi-recursive evolution.
    
    Parameters:
    -----------
    scaling_factors : List[float] or None
        List of scaling factors to analyze.
    num_qubits : int
        Number of qubits in the system.
    state_label : str
        Label for the initial state.
    hamiltonian_type : str
        Type of Hamiltonian to use.
    n_steps : int
        Number of evolution time steps.
    recursion_depth : int
        Recursion depth for phi-recursive evolution.
        
    Returns:
    --------
    Dict
        Dictionary containing analysis results.
    """
    # Use default scaling factors if not provided
    if scaling_factors is None:
        phi = PHI
        scaling_factors = np.sort(np.unique(np.concatenate([
            np.linspace(0.5, 3.0, 25),
            [phi]
        ])))
    
    # Initialize results
    results = {
        'standard_results': {},
        'phi_recursive_results': {},
        'comparative_metrics': {},
        'statistical_significance': {}
    }
    
    # Find phi index
    phi = PHI
    phi_idx = np.argmin(np.abs(scaling_factors - phi))
    
    # Initialize arrays for statistical analysis
    state_overlaps = []
    dimension_differences = []
    
    # Run analysis for each scaling factor
    for factor in tqdm(scaling_factors, desc="Running comparative analysis", unit="factor"):
        # Run standard evolution
        std_result = run_state_evolution_fixed(
            num_qubits=num_qubits,
            state_label=state_label,
            hamiltonian_type=hamiltonian_type,
            n_steps=n_steps,
            scaling_factor=factor
        )
        
        # Run phi-recursive evolution
        phi_result = run_phi_recursive_evolution_fixed(
            num_qubits=num_qubits,
            state_label=state_label,
            hamiltonian_type=hamiltonian_type,
            n_steps=n_steps,
            scaling_factor=factor,
            recursion_depth=recursion_depth
        )
        
        # Compute comparative metrics
        state_overlap = abs(std_result.states[-1].overlap(phi_result.states[-1]))**2
        
        # Consistency check: both states should be similar for scaling_factor=1.0
        if abs(factor - 1.0) < 1e-10:
            if abs(state_overlap - 1.0) > 1e-6:
                print(f"Warning: States should be identical at scaling_factor=1.0, but overlap = {state_overlap:.6f}")
        
        # Calculate phi proximity (for reference only)
        phi_proximity = np.exp(-(factor - phi)**2 / 0.1)
        
        # Store results
        results['standard_results'][factor] = std_result
        results['phi_recursive_results'][factor] = phi_result
        
        # Store metrics
        metrics = {
            'state_overlap': state_overlap,
            'phi_proximity': phi_proximity
        }
        
        # Add fractal dimension comparison if possible
        # We need to make sure both values are available and not None
        std_dim = None
        phi_dim = None
        
        if hasattr(std_result, 'fractal_dimensions') and len(std_result.fractal_dimensions) > 0:
            std_dim = np.nanmean(std_result.fractal_dimensions)
        
        if hasattr(phi_result, 'phi_dimension') and phi_result.phi_dimension is not None:
            phi_dim = phi_result.phi_dimension
            
        if std_dim is not None and phi_dim is not None:
            metrics['dimension_difference'] = phi_dim - std_dim
            dimension_differences.append(phi_dim - std_dim)
        
        results['comparative_metrics'][factor] = metrics
        state_overlaps.append(state_overlap)
    
    # Perform statistical significance analysis
    # Check if phi shows significantly different behavior
    state_overlaps = np.array(state_overlaps)
    dimension_differences = np.array(dimension_differences) if dimension_differences else np.array([])
    
    # Calculate statistics for state overlap
    if len(state_overlaps) > 0:
        from scipy.stats import ttest_1samp
        
        # Calculate for state overlap
        non_phi_overlaps = np.concatenate([
            state_overlaps[:phi_idx],
            state_overlaps[phi_idx+1:]
        ])
        
        if len(non_phi_overlaps) > 0:
            phi_overlap = state_overlaps[phi_idx]
            t_stat, p_value = ttest_1samp(non_phi_overlaps, phi_overlap)
            
            # Calculate z-score
            mean = np.mean(non_phi_overlaps)
            std = np.std(non_phi_overlaps)
            z_score = (phi_overlap - mean) / std if std > 0 else 0.0
            
            results['statistical_significance']['state_overlap'] = {
                'p_value': float(p_value),
                'z_score': float(z_score),
                'significant': float(p_value) < 0.05,
                'sample_size': len(non_phi_overlaps)
            }
        
        # Calculate for dimension differences if available
        if len(dimension_differences) > 0:
            non_phi_differences = np.concatenate([
                dimension_differences[:phi_idx],
                dimension_differences[phi_idx+1:]
            ])
            
            if len(non_phi_differences) > 0:
                phi_difference = dimension_differences[phi_idx]
                t_stat, p_value = ttest_1samp(non_phi_differences, phi_difference)
                
                # Calculate z-score
                mean = np.mean(non_phi_differences)
                std = np.std(non_phi_differences)
                z_score = (phi_difference - mean) / std if std > 0 else 0.0
                
                results['statistical_significance']['dimension_difference'] = {
                    'p_value': float(p_value),
                    'z_score': float(z_score),
                    'significant': float(p_value) < 0.05,
                    'sample_size': len(non_phi_differences)
                }
    
    return results

def simulate_noise_evolution(H, psi0, times, c_ops):
    """
    Simulate quantum evolution with noise (collapse operators).
    
    Parameters:
    -----------
    H : Qobj
        System Hamiltonian (already properly scaled).
    psi0 : Qobj
        Initial quantum state.
    times : array
        Array of time points for the evolution.
    c_ops : list
        List of collapse operators (noise operators).
        
    Returns:
    --------
    EvolutionResult
        Result of the quantum evolution with noise.
    """
    from qutip import mesolve
    
    print(f"Simulating noise evolution with {len(c_ops)} collapse operators...")
    
    # Run the evolution with noise using QuTiP's solver
    result = mesolve(H, psi0, times, c_ops=c_ops)
    
    # Create and return our standard result format
    evolution_result = EvolutionResult(
        states=result.states,
        times=times,
        scaling_factor=1.0,  # We don't track scaling factor here since H is already scaled
        params={
            'num_qubits': psi0.dims[0][0],  # Extract from state dimensions
            'noise_operators': len(c_ops)
        }
    )
    
    return evolution_result

# For demonstration and testing
if __name__ == "__main__":
    # Run a simple test
    print(f"Testing fixed evolution implementation with phi = {PHI:.6f}")
    
    # Run standard evolution
    std_result = run_state_evolution_fixed(num_qubits=1, scaling_factor=1.0)
    
    # Run phi-recursive evolution
    phi_result = run_phi_recursive_evolution_fixed(num_qubits=1, scaling_factor=PHI)
    
    # Calculate overlap
    overlap = abs(std_result.states[-1].overlap(phi_result.states[-1]))**2
    print(f"Final state overlap: {overlap:.6f}")
    
    print("Fixed evolution implementation test complete.")

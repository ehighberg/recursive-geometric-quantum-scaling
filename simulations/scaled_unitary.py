"""
Scaled unitary operator implementation using qutip's features.

This module provides implementations for:
- Linear scaling of unitary operators
- Non-linear phi-resonant scaling
- Recursive geometric scaling with golden ratio properties

All functions are designed to avoid arbitrary parameter values and provide
configurable parameters with physically-justified defaults.
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

def _calculate_phi_proximity(scaling_factor, gaussian_width=None):
    """
    Calculate a smooth proximity measure to the golden ratio.
    
    This function creates a Gaussian bell curve centered at φ, giving a
    continuous measure of how close a value is to φ.
    
    Parameters:
    -----------
    scaling_factor (float): The value to check against φ
    gaussian_width (float, optional): Width parameter for the Gaussian
    
    Returns:
    --------
    float: Proximity measure in range [0,1] where 1 means exactly φ
    """
    if gaussian_width is None:
        gaussian_width = PHI_GAUSSIAN_WIDTH
        
    phi = PHI  # Golden ratio from constants
    
    # Avoid division by zero if gaussian_width is too small
    if gaussian_width <= 1e-10:
        gaussian_width = 1e-10
        
    # Calculate Gaussian centered at phi
    proximity = np.exp(-((scaling_factor - phi)**2) / gaussian_width)
    
    return proximity

# These functions are the canonical implementations for scaled and phi-recursive unitaries.
# They are used by the ScaledCircuit class in quantum_circuit.py.
def get_scaled_unitary(H, time, scaling_factor=1.0):
    """
    Get the linearly scaled unitary operator for a given Hamiltonian.
    
    This uses matrix logarithm and exponentiation to perform continuous scaling
    of a unitary operator: U^s = exp(s * log(U)) where s is scaling_factor.
    
    Parameters:
    -----------
    H (Qobj): Hamiltonian operator
    time (float): Evolution time
    scaling_factor (float): Factor to scale the unitary
    
    Returns:
    --------
    Qobj: Scaled unitary operator
    
    Notes:
    ------
    The mathematical validity of this method depends on the logarithm being 
    well-defined for the unitary operator. For scaling factors close to 1.0, 
    the approximation is highly accurate.
    """
    # Get unitary for original Hamiltonian
    U = propagator(H, time)
    
    # Special case: scaling_factor = 1.0 returns the original unitary
    if scaling_factor == 1.0:
        return U
    else:
        # For non-unity scaling, use matrix logarithm and exponentiation
        logU = U.logm()  # Matrix logarithm
        scaled_U = (scaling_factor * logU).expm()
        
        # Ensure the result preserves unitarity (within numerical precision)
        unitarity_check = (scaled_U * scaled_U.dag())
        dims = unitarity_check.dims
        I = Qobj(np.eye(scaled_U.shape[0]), dims=dims)
        if not np.allclose((scaled_U * scaled_U.dag()).full(), I.full(), 
                           rtol=UNITARITY_RTOL, atol=UNITARITY_ATOL):
            # Apply re-normalization to ensure unitarity
            scaled_U = _ensure_unitarity(scaled_U)
        
        return scaled_U


def get_phi_recursive_unitary(H, time, scaling_factor=1.0, recursion_depth=3,
                             gaussian_width=None, phi_threshold=None, 
                             correction_cutoff=None):
    """
    Create a unitary with recursive golden-ratio-based structure by implementing
    the recursion relation U_φ(t) = U(t/φ) · U(t/φ²), which creates self-similar
    patterns in the quantum evolution.
    
    Mathematical model:
    U_φ(t) = U(t/φ) · U(t/φ²)
    
    where U(t) = exp(-i·H·t·scaling_factor)
    
    Parameters:
    -----------
    H : Qobj
        Hamiltonian operator
 (unscaled)
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
    
   #: The scaling factor is applied ONCE here
        H_scaled = scaling_factor * H
        return (-1j * H_scaled * time).expm()
    
    # Recursive case: implement the mathematical relation U_φ(t) = U(t/φ) · U(t/φ²)
    # Apply recursion with proper parameter passing:
    # - Pass the SAME scaling_factor down the recursion chain
    # - Only modify the time parameter with phi divisions
    U_phi1 = get_phi_recursive_unitary(
H, time/PHI, scaling_factor, recursion_depth-1)
    U_phi2 = get_phi_recursive_unitary(
H, time/(PHI**2), scaling_factor, recursion_depth-1)
    
    # Combine recursive unitaries
    return U_phi1 * U_phi2

def simulate_scaled_unitary(scaling_factor=1.0):
    """
    Simulates the scaling of a unitary operator in a quantum system.
    
    Parameters:
    - scaling_factor (float): Factor by which to scale the unitary operator
    
    Returns:
    - U (Qobj): Original unitary evolution operator
    - U_scaled (Qobj): Scaled unitary evolution operator
    """
    # Example Hamiltonian
    H = 2 * np.pi * 0.1 * sigmax()
    time = 10.0
    
    # Get original and scaled unitaries
    U = get_scaled_unitary(H, time, 1.0)
    U_scaled = get_scaled_unitary(H, time, scaling_factor)
    
    return U, U_scaled

def create_scaled_gate(H, time, scaling_factor=1.0):
    """
    Create a custom gate from a scaled unitary operator.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    
    Returns:
    - Gate: Custom gate with scaled unitary
    """
    U_scaled = get_scaled_unitary(H, time, scaling_factor)
    
    # Create custom gate
    gate = Gate(name=f"SCALED_{scaling_factor}", targets=[0])
    gate.matrix = U_scaled.full()
    
    return gate

def get_scaled_circuit(H, time, scaling_factors):
    """
    Create a circuit with multiple scaled unitary operations.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factors (list): List of scaling factors
    
    Returns:
    - QubitCircuit: Circuit with scaled operations
    """
    # Create circuit
    qc = QubitCircuit(1)
    
    # Add gates for each scaling factor
    for factor in scaling_factors:
        gate = create_scaled_gate(H, time, factor)
        qc.add_gate(gate)
    
    return qc

def simulate_scaled_sequence(H, time, scaling_factors, initial_state):
    """
    Simulate a sequence of scaled unitary operations.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factors (list): List of scaling factors
    - initial_state (Qobj): Initial quantum state
    
    Returns:
    - list: List of states after each operation
    """
    # Create circuit
    qc = get_scaled_circuit(H, time, scaling_factors)
    
    # Initialize state list
    states = [initial_state]
    current_state = initial_state
    
    # Apply each scaled operation
    for factor in scaling_factors:
        U = get_scaled_unitary(H, time, factor)
        current_state = U * current_state
        states.append(current_state)
    
    return states

def analyze_scaling_properties(H, time, scaling_factors):
    """
    Analyze how properties of the unitary operator change with scaling.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factors (list): List of scaling factors to analyze
    
    Returns:
    - dict: Dictionary containing analysis results
    """
    results = {
        'unitaries': [],
        'traces': [],
        'eigenvalues': [],
        'determinants': []
    }
    
    for factor in scaling_factors:
        U = get_scaled_unitary(H, time, factor)
        results['unitaries'].append(U)
        results['traces'].append(U.tr())
        results['eigenvalues'].append(U.eigenenergies())
        results['determinants'].append(U.det())

    return results


def analyze_phi_recursion_properties(H, time, scaling_factors, recursion_depths=None):
    """
    Analyze how properties change with phi-recursive scaling.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factors (list): List of scaling factors to analyze
    - recursion_depths (list): List of recursion depths to analyze (default [1,2,3])
    
    Returns:
    - dict: Dictionary containing analysis results
    """
    if recursion_depths is None:
        recursion_depths = [1, 2, 3]
    
    results = {
        'scaling_factors': scaling_factors,
        'recursion_depths': recursion_depths,
        'unitaries': {},
        'traces': {},
        'eigenvalues': {},
        'phi_sensitivity': {}
    }
    
    # Compute phi sensitivity metric
    phi = PHI
    phi_distances = [abs(factor - phi) for factor in scaling_factors]
    results['phi_distances'] = phi_distances
    
    # Analyze unitaries at different recursion depths
    for depth in recursion_depths:
        results['unitaries'][depth] = []
        results['traces'][depth] = []
        results['eigenvalues'][depth] = []
        results['phi_sensitivity'][depth] = []
        
        for factor in scaling_factors:
            # Get phi-recursive unitary
            U = get_phi_recursive_unitary(H, time, factor, depth)
            
            # Store basic properties
            results['unitaries'][depth].append(U)
            results['traces'][depth].append(U.tr())
            results['eigenvalues'][depth].append(U.eigenenergies())
            
            # Calculate phi-sensitivity metric
            # (Higher value means more sensitive to being at phi)
            phi_sensitivity = np.abs(U.tr() - get_scaled_unitary(H, time, factor).tr())
            results['phi_sensitivity'][depth].append(float(phi_sensitivity))
    
    return results
 

def create_phi_resonant_gate(H, time, scaling_factor=PHI, recursion_depth=2):
    """
    Create a custom gate with phi-resonant properties.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    - recursion_depth (int): Depth of recursion for phi-based patterns
    
    Returns:
    - Gate: Custom gate with phi-resonant unitary
    """
    # Get phi-recursive unitary
    U_phi = get_phi_recursive_unitary(H, time, scaling_factor, recursion_depth)
    
    # Create custom gate
    gate = Gate(name=f"PHI_RESONANT_{scaling_factor:.4f}_d{recursion_depth}", targets=[0])
    gate.matrix = U_phi.full()
    
    return gate


def get_phi_resonant_circuit(H, time, num_qubits=1, depth=3):
    """
    Create a circuit with phi-resonant properties.
    
    Parameters:
    - H (Qobj): Single-qubit Hamiltonian operator
    - time (float): Evolution time
    - num_qubits (int): Number of qubits in the circuit
    - depth (int): Depth of the circuit (layers)
    
    Returns:
    - QubitCircuit: Circuit with phi-resonant operations
    """
    # Create circuit with the specified number of qubits
    qc = QubitCircuit(num_qubits)
    
    # Create Fibonacci sequence for gate placement
    fib = [1, 1]
    while len(fib) < depth + 2:
        fib.append(fib[-1] + fib[-2])
    
    # Generate phi-derived scaling factors
    phi = PHI
    scaling_factors = [phi**(i % 3) for i in range(depth)]
    
    # Add gates in a Fibonacci pattern
    for i in range(min(depth, num_qubits)):
        for j in range(num_qubits):
            # Add gates at positions based on Fibonacci sequence
            if (j + i) % fib[min(i+1, len(fib)-1)] == 0:
                # For single qubit gates
                gate = create_phi_resonant_gate(
                    H, 
                    time,
                    scaling_factor=scaling_factors[i],
                    recursion_depth=min(i+1, 3)
                )
                gate.targets = [j]
                qc.add_gate(gate)
                
                # For two-qubit gates, if possible
                if j < num_qubits - i:
                    # Create control gate between qubits
                    control_idx = (j + fib[min(i, len(fib)-1)]) % num_qubits
                    qc.add_gate("CNOT", controls=j, targets=[control_idx])
    
    return qc


def simulate_phi_sequence(H, time, initial_state, recursion_depth=2):
    """
    Simulate an evolution sequence with phi-recursive unitaries.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - initial_state (Qobj): Initial quantum state
    - recursion_depth (int): Depth of recursion for phi-based patterns
    
    Returns:
    - dict: Dictionary containing simulation results
    """
    # Define range of scaling factors with phi for comparison
    phi = PHI
    # Include phi and nearby values for comparison
    scaling_factors = [0.5, 1.0, 1.5, phi, 2.0, 2.5, 3.0]
    
    results = {
        'scaling_factors': scaling_factors,
        'states': {},
        'expectation_values': {},
        'phi_sensitivity': []
    }
    
    # Analyze expectation value of sigmaz
    sz = sigmaz()
    
    # Simulate for each scaling factor
    for factor in scaling_factors:
        # Get phi-recursive unitary
        U = get_phi_recursive_unitary(H, time, factor, recursion_depth)
        
        # Apply to initial state
        final_state = U * initial_state
        
        # Store results
        results['states'][factor] = final_state
        results['expectation_values'][factor] = np.real((final_state.dag() * sz * final_state).tr())
        
        # Compare to standard scaling 
        std_final_state = get_scaled_unitary(H, time, factor) * initial_state
        state_diff = (final_state - std_final_state).norm()
        results['phi_sensitivity'].append(float(state_diff))
    
    return results

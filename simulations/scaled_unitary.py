"""
Scaled unitary operator implementation using qutip's features.

This module includes implementations for:
- Linear scaling of unitary operators
- Non-linear phi-resonant scaling
- Recursive geometric scaling with golden ratio properties
"""

import numpy as np
from qutip import Qobj, propagator, sigmax, sigmaz
from qutip.solver import Result
from qutip_qip.operations import Gate
from qutip_qip.circuit import QubitCircuit
from constants import PHI

#TODO: use these functions instead of existing code with the same purpose, or remove them
def get_scaled_unitary(H, time, scaling_factor=1.0):
    """
    Get the linearly scaled unitary operator for a given Hamiltonian.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    
    Returns:
    - Qobj: Scaled unitary operator
    """
    # Get unitary for original Hamiltonian
    U = propagator(H, time)
    
    # Scale the unitary using matrix exponentiation
    # U^s = exp(s * log(U)) where s is the scaling factor
    if scaling_factor == 1.0:
        return U
    else:
        logU = U.logm()  # Matrix logarithm
        return (scaling_factor * logU).expm()


def get_phi_recursive_unitary(H, time, scaling_factor=1.0, recursion_depth=3):
    """
    Create a unitary with recursive golden-ratio-based structure.
    
    Parameters:
    - H (Qobj): Hamiltonian operator
    - time (float): Evolution time
    - scaling_factor (float): Factor to scale the unitary
    - recursion_depth (int): Depth of recursion for phi-based patterns
    
    Returns:
    - Qobj: Scaled unitary with potential phi-resonance
    """
    # Use matrix exponentiation directly instead of propagator to avoid integration issues
    U_base = (-1j * H * time).expm()
    
    # Base case for recursion
    if recursion_depth <= 0 or scaling_factor == 1.0:
        return U_base
    
    # Create recursive structure based on Fibonacci/golden ratio pattern
    phi = PHI  # Golden ratio
    
    # Calculate proximity to phi for resonance effects
    phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
    
    if phi_proximity > 0.9:  # Close to phi
        # At phi, create recursive operator structure 
        U_phi1 = get_phi_recursive_unitary(H, time/phi, scaling_factor/phi, recursion_depth-1)
        U_phi2 = get_phi_recursive_unitary(H, time/phi**2, scaling_factor/phi**2, recursion_depth-1)
        return U_phi1 * U_phi2
    else:
        # For non-phi values, use different composition rule
        logU = U_base.logm()
        U_scaled = (scaling_factor * logU).expm()
        
        # Add non-linear term that's most significant near phi
        correction_factor = scaling_factor/(1 + abs(scaling_factor - phi))
        correction = (correction_factor * logU**2).expm() if correction_factor > 0.2 else Qobj(np.eye(U_base.shape[0]))
        
        return U_scaled * correction

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
    gate = Gate(name=f"PHI_RESONANT_{scaling_factor:.3f}_d{recursion_depth}", targets=[0])
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

"""
Scaled unitary operator implementation using qutip's features.
"""

import numpy as np
from qutip import Qobj, propagator, sigmax, sigmaz
from qutip_qip.operations import Gate
from qutip_qip.circuit import QubitCircuit

#TODO: use these functions instead of existing code with the same purpose, or remove them
def get_scaled_unitary(H, time, scaling_factor=1.0):
    """
    Get the scaled unitary operator for a given Hamiltonian.
    
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
    
    # Get total unitary
    U_total = qc.compute_unitary()
    
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

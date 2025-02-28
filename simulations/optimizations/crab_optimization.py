"""
CRAB (Chopped RAndom Basis) optimization implementation for quantum circuits.
"""

import numpy as np
from qutip import propagator
from scipy.optimize import minimize

def crab_basis_functions(t, n_freq=5):
    """Generate CRAB basis functions."""
    basis = []
    for k in range(1, n_freq + 1):
        # Use both sine and cosine functions for each frequency
        basis.append(np.sin(2 * np.pi * k * t))
        basis.append(np.cos(2 * np.pi * k * t))
    return np.array(basis)

def apply_crab_optimization(circuit, target_unitary):
    """
    Apply CRAB optimization to a quantum circuit using direct pulse evolution.
    
    This implementation uses the Chopped RAndom Basis (CRAB) approach to optimize
    control pulses by expanding them in a truncated basis of trigonometric functions.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to optimize.
        target_unitary (Qobj): The target unitary operator.
        
    Returns:
        np.ndarray: Optimized control pulse amplitudes.
    """
    # Define the drift Hamiltonian
    H_d = circuit.base_hamiltonian
    
    # Define the control Hamiltonians
    H_c = [circuit.circuit.qubits[0].ops['x'], circuit.circuit.qubits[1].ops['x']]
    
    # Setup time grid
    n_ts = 100
    evo_time = circuit.total_time if hasattr(circuit, 'total_time') else 5.0
    tlist = np.linspace(0, evo_time, n_ts)
    
    # Create total Hamiltonian with drift and control terms
    def create_hamiltonian(control_amps):
        H_total = H_d
        for h, amp in zip(H_c, control_amps):
            H_total += h * amp.mean()  # Use mean amplitude for simplicity
        return H_total
    
    # Generate basis functions
    basis_funcs = crab_basis_functions(tlist / evo_time)
    n_basis = basis_funcs.shape[0]
    
    def objective_function(x):
        """Calculate infidelity using CRAB basis expansion."""
        # Reshape parameters into coefficients for each control and basis function
        coeffs = x.reshape(len(H_c), n_basis)
        
        # Construct control amplitudes using basis expansion
        control_amps = []
        for ctrl_coeffs in coeffs:
            amp = np.sum(ctrl_coeffs[:, np.newaxis] * basis_funcs, axis=0)
            control_amps.append(amp)
        
        # Create Hamiltonian with current coefficients and get evolved unitary
        H_total = create_hamiltonian(control_amps)
        U = propagator(H_total, tlist[-1])
        
        # Calculate fidelity
        fidelity = abs((target_unitary.dag() * U).tr()) / target_unitary.shape[0]
        return 1.0 - fidelity
    
    # Initial guess for basis coefficients
    x0 = np.random.randn(len(H_c) * n_basis) * 0.1
    
    # Run optimization
    result = minimize(
        objective_function,
        x0,
        method='Nelder-Mead',  # CRAB typically uses derivative-free optimization
        options={
            'maxiter': 1000,
            'xatol': 1e-6,
            'fatol': 1e-6
        }
    )
    
    # Convert optimal basis coefficients back to time-domain pulses
    final_coeffs = result.x.reshape(len(H_c), n_basis)
    optimal_amps = []
    for ctrl_coeffs in final_coeffs:
        amp = np.sum(ctrl_coeffs[:, np.newaxis] * basis_funcs, axis=0)
        optimal_amps.append(amp)
    
    return np.array(optimal_amps)

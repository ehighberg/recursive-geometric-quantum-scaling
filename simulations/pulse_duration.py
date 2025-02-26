"""
Pulse duration optimization using qutip's control optimization features.
"""

import numpy as np
from qutip import Qobj, sesolve, mesolve, sigmax, sigmaz
from qutip import basis, sigmay  # Used only in specific functions
from qutip_qip.pulse import Pulse
from qutip_qip.circuit import QubitCircuit

def optimize_pulse_duration(target_unitary, max_time=10.0, num_tslots=100, noise_rate=0.0):
    """
    Optimize pulse duration for a target unitary operation using direct pulse evolution.
    
    Parameters:
    - target_unitary (Qobj): Target unitary operation
    - max_time (float): Maximum evolution time
    - num_tslots (int): Number of time slots for optimization
    - noise_rate (float): Rate of noise to include in optimization
    
    Returns:
    - result: Dictionary containing optimized pulse parameters and fidelity
    """
    # System Hamiltonian
    H_d = sigmaz()  # Drift Hamiltonian
    H_c = [sigmax(), sigmay()]  # Control Hamiltonians
    
    # Time slots
    tlist = np.linspace(0, max_time, num_tslots)
    
    # Create initial coefficients (constant amplitude)
    coeffs = [np.ones(num_tslots) for _ in H_c]
    
    # Create pulse with initial constant amplitudes
    pulse = Pulse(
        H_d,  # Target Hamiltonian
        [0],  # Target qubits (as positional argument)
        tlist=tlist,  # Time points
        coeff=coeffs  # Control coefficients
    )
    
    # Create QobjEvo for evolution
    H_total = H_d
    for h_c, coeff in zip(H_c, coeffs):
        H_total += h_c * coeff[0]  # Use first coefficient as initial
    
    # Include noise if specified
    c_ops = []
    if noise_rate > 0:
        c_ops = [np.sqrt(noise_rate) * sigmax()]
    
    # Evolve the system
    result = sesolve(H_total, basis(2, 0), tlist, c_ops=c_ops)
    U = result.states[-1]
    
    # Calculate fidelity
    fidelity = abs((target_unitary.dag() * U).tr()) / target_unitary.shape[0]
    
    return {
        'final_unitary': U,
        'fidelity': fidelity,
        'pulse': pulse,
        'coeffs': coeffs,
        'tlist': tlist
    }

def simulate_pulse_duration(pulse_rate=0.1, noise_rate=0.0):
    """
    Simulates the pulse duration for a quantum system.
    
    Parameters:
    - pulse_rate (float): The rate of the pulse
    - noise_rate (float): The rate of noise to include in the simulation
    
    Returns:
    - times (numpy.ndarray): Array of time points
    - expectation_values (numpy.ndarray): Expectation values of sigmax over time
    """
    # System parameters
    H = 2 * np.pi * pulse_rate * sigmax()
    psi0 = basis(2, 0)
    times = np.linspace(0.0, 10.0, 100)
    
    if noise_rate > 0:
        # Include noise in simulation
        c_ops = [np.sqrt(noise_rate) * sigmax()]
        result = mesolve(H, psi0, times, c_ops, e_ops=[sigmax()])
    else:
        # Noise-free simulation
        result = sesolve(H, psi0, times, e_ops=[sigmax()])
    
    return times, result.expect[0]

def get_optimal_duration(target_gate, max_time=10.0, num_tslots=100, fidelity_threshold=0.99):
    """
    Find the optimal duration for implementing a target gate.
    
    Parameters:
    - target_gate (str): Name of target gate ("X", "Y", "Z", "H", etc.)
    - max_time (float): Maximum evolution time to consider
    - num_tslots (int): Number of time slots for optimization
    - fidelity_threshold (float): Target fidelity threshold
    
    Returns:
    - float: Optimal duration
    - dict: Optimization results
    """
    # Create circuit with target gate
    qc = QubitCircuit(1)
    qc.add_gate(target_gate, targets=[0])
    
    # Get target unitary
    target_U = qc.compute_unitary()
    
    # Try different durations
    durations = np.linspace(0.1, max_time, 20)
    results = []
    
    for T in durations:
        result = optimize_pulse_duration(target_U, max_time=T, num_tslots=num_tslots)
        results.append({
            'duration': T,
            'fidelity': result['fidelity'],
            'pulses': result['coeffs']
        })
        
        current_fidelity = result['fidelity']
        if current_fidelity >= fidelity_threshold:
            break
    
    # Find best result
    best_result = max(results, key=lambda x: x['fidelity'])
    
    return best_result['duration'], results

def analyze_duration_scaling(target_gate, scaling_factors, base_duration=1.0):
    """
    Analyze how pulse duration scales with different parameters.
    
    Parameters:
    - target_gate (str): Name of target gate
    - scaling_factors (list): List of scaling factors to analyze
    - base_duration (float): Base duration for scaling analysis
    
    Returns:
    - dict: Analysis results
    """
    qc = QubitCircuit(1)
    qc.add_gate(target_gate, targets=[0])
    target_U = qc.compute_unitary()
    
    results = {
        'durations': [],
        'fidelities': [],
        'pulse_shapes': []
    }
    
    for factor in scaling_factors:
        # Scale duration
        T = base_duration * factor
        
        # Optimize pulse for scaled duration
        result = optimize_pulse_duration(target_U, max_time=T)
        
        results['durations'].append(T)
        results['fidelities'].append(result['fidelity'])
        results['pulse_shapes'].append(result['coeffs'])
    
    return results

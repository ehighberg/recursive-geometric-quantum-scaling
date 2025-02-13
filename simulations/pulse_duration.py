"""
Pulse duration optimization using qutip's control optimization features.
"""

import numpy as np
from qutip import Qobj, sesolve, mesolve, basis, sigmax, sigmay, sigmaz
from qutip.control import optimize_pulse_unitary
from qutip_qip.compiler import GateCompiler
from qutip_qip.circuit import QubitCircuit

def optimize_pulse_duration(target_unitary, max_time=10.0, num_tslots=100, noise_rate=0.0):
    """
    Optimize pulse duration for a target unitary operation.
    
    Parameters:
    - target_unitary (Qobj): Target unitary operation
    - max_time (float): Maximum evolution time
    - num_tslots (int): Number of time slots for optimization
    - noise_rate (float): Rate of noise to include in optimization
    
    Returns:
    - result: Optimization result containing optimal pulses
    """
    # System Hamiltonian
    H_d = sigmaz()  # Drift Hamiltonian
    H_c = [sigmax(), sigmay()]  # Control Hamiltonians
    
    # Initial pulse guess (constant amplitude)
    initial_pulses = [np.ones(num_tslots) for _ in H_c]
    
    # Time slots
    tslots = np.linspace(0, max_time, num_tslots)
    
    # Optimize pulse
    if noise_rate > 0:
        # Include noise operators in optimization
        c_ops = [np.sqrt(noise_rate) * sigmax()]
        result = optimize_pulse_unitary(
            H_d, H_c, target_unitary,
            num_tslots=num_tslots,
            evo_time=max_time,
            initial_pulses=initial_pulses,
            c_ops=c_ops
        )
    else:
        result = optimize_pulse_unitary(
            H_d, H_c, target_unitary,
            num_tslots=num_tslots,
            evo_time=max_time,
            initial_pulses=initial_pulses
        )
    
    return result

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
        fidelity = result.fidelity
        results.append({
            'duration': T,
            'fidelity': fidelity,
            'pulses': result.final_amps
        })
        
        if fidelity >= fidelity_threshold:
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
        results['fidelities'].append(result.fidelity)
        results['pulse_shapes'].append(result.final_amps)
    
    return results

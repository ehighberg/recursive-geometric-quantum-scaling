#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py

"""
Circuit-based approach using qutip-qip features for multi-qubit and braiding operations.
"""

import sys
import os
# Add the project root to the Python path to ensure modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qutip import sigmaz, sigmax, qeye, tensor, basis, Options
from qutip_qip.circuit import QubitCircuit
from simulations.quantum_state import state_zero, fib_anyon_state_2d
from simulations.quantum_circuit import StandardCircuit, PhiScaledCircuit, FibonacciBraidingCircuit
from simulations.amplitude_scaling import get_pulse_sequence

def run_standard_twoqubit_circuit(noise_config=None):
    """
    2-qubit uniform approach with configurable noise.
    H0 = sigma_z(1) + 0.1 sigma_x(2)
    
    Parameters:
    - noise_config (dict): Optional noise configuration
    
    Returns:
    - result: Evolution result containing states and times
    """
    # Create base Hamiltonian
    H0 = tensor(sigmaz(), qeye(2)) + 0.1 * tensor(qeye(2), sigmax())
    
    # Create circuit
    circ = StandardCircuit(H0, total_time=5.0, n_steps=50)
    
    # Add some standard gates
    qc = QubitCircuit(2)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("RX", targets=[1], arg_value=0.5)
    
    # Get pulse sequence for the gates
    H_list, coeff_list, tlist = get_pulse_sequence(H0, total_time=5.0, steps=50)
    
    # Initialize state
    psi_init = state_zero(num_qubits=2)
    
    # Evolve with or without noise
    if noise_config:
        result = circ.evolve_open(psi_init)
    else:
        result = circ.evolve_closed(psi_init)
    
    return result

def run_phi_scaled_twoqubit_circuit(scaling_factor=1.0, noise_config=None):
    """
    2-qubit phi-scaled approach with configurable noise.
    H0 = sigma_z(1) + 0.5 sigma_x(2)
    
    Parameters:
    - scaling_factor (float): Scaling factor for evolution
    - noise_config (dict): Optional noise configuration
    
    Returns:
    - result: Evolution result containing states and times
    """
    # Create base Hamiltonian
    H0 = tensor(sigmaz(), qeye(2)) + 0.5 * tensor(qeye(2), sigmax())
    
    # Create circuit with scaling
    pcirc = PhiScaledCircuit(H0, scaling_factor=scaling_factor)
    
    # Add scaled gates
    qc = QubitCircuit(2)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("RX", targets=[1], arg_value=0.5 * scaling_factor)
    
    # Initialize state
    psi_init = state_zero(num_qubits=2)
    
    # Evolve with or without noise
    if noise_config:
        result = pcirc.evolve_open(psi_init, n_steps=5)
    else:
        result = pcirc.evolve_closed(psi_init, n_steps=5)
    
    return result

def run_fibonacci_braiding_circuit():
    # TODO: Implement Fibonacci braiding circuit using variable parameters from app
    """
    Fibonacci anyon braiding circuit in 2D subspace.
    Uses B1, B2 braid operators with qutip-qip gate compilation.
    
    Returns:
    - psi_final: Final state after braiding operations
    """
    from simulations.scripts.fibonacci_anyon_braiding import braid_b1_2d, braid_b2_2d
    
    # Get braid operators
    B1_2 = braid_b1_2d()
    B2_2 = braid_b2_2d()
    
    # Create braiding circuit
    fib_circ = FibonacciBraidingCircuit()
    
    # Add braids as custom gates
    fib_circ.add_braid(B1_2)
    fib_circ.add_braid(B2_2)
    
    # Initialize state and evolve
    psi_init = fib_anyon_state_2d()
    psi_final = fib_circ.evolve(psi_init)
    
    return psi_final

def analyze_circuit_noise_effects(circuit_type="standard", noise_rates=None):
    """
    Analyze how different noise types affect circuit evolution.
    
    Parameters:
    - circuit_type (str): "standard" or "phi_scaled"
    - noise_rates (list): List of noise rates to test
    
    Returns:
    - dict: Analysis results
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.01, 0.05, 0.1]
    
    results = {
        'fidelities': [],
        'purities': [],
        'noise_rates': noise_rates
    }
    
    # Initial state for comparison
    psi_init = state_zero(num_qubits=2)
    
    for rate in noise_rates:
        # Configure noise
        noise_config = {
            'relaxation': rate,
            'dephasing': rate,
            'thermal': 0.0,
            'measurement': 0.0
        }
        
        # Run circuit with noise
        if circuit_type == "standard":
            result = run_standard_twoqubit_circuit(noise_config=noise_config)
        else:
            result = run_phi_scaled_twoqubit_circuit(noise_config=noise_config)
        
        # Calculate fidelity with initial state
        final_state = result.states[-1]
        fidelity = (psi_init.dag() * final_state * psi_init).tr().real
        
        # Calculate purity
        purity = (final_state * final_state).tr().real
        
        results['fidelities'].append(fidelity)
        results['purities'].append(purity)
    
    return results

def run_quantum_gate_circuit(circuit_type="Single Qubit", optimization=None, noise_config=None):
    """
    Run quantum circuit with specified gate operations.
    
    Parameters:
    - circuit_type (str): Type of circuit ("Single Qubit", "CNOT", "Toffoli", "Custom")
    - optimization (str): Optimization method ("GRAPE", "CRAB", "None")
    - noise_config (dict): Optional noise configuration
    
    Returns:
    - result: Evolution result containing states and times
    """
    if circuit_type == "Single Qubit":
        # Create base Hamiltonian for single qubit
        H0 = sigmaz()
        
        # Create circuit
        circ = StandardCircuit(H0, total_time=5.0, n_steps=50)
        
        # Add single qubit gates
        qc = QubitCircuit(1)
        qc.add_gate("RX", targets=[0], arg_value=0.5)
        qc.add_gate("RY", targets=[0], arg_value=0.3)
        
        # Initialize state
        psi_init = basis([2], 0)
        
    elif circuit_type == "CNOT":
        # Use existing two-qubit circuit
        return run_standard_twoqubit_circuit(noise_config=noise_config)
        
    elif circuit_type == "Toffoli":
        raise NotImplementedError("Toffoli gate not yet implemented")
        
    else:  # Custom
        raise NotImplementedError("Custom circuits not yet implemented")
    
    # Apply optimization if specified
    if optimization and optimization != "None":
        if optimization == "GRAPE":
            # Add GRAPE optimization
            opts = Options(max_step=1000, accuracy_factor=1e-3)
            # ... implement GRAPE optimization ...
            pass
        elif optimization == "CRAB":
            # Add CRAB optimization
            # ... implement CRAB optimization ...
            pass
        
    # Evolve with or without noise
    if noise_config:
        result = circ.evolve_open(psi_init)
    else:
        result = circ.evolve_closed(psi_init)
    
    return result

if __name__ == "__main__":
    # Example usage
    res_std = run_standard_twoqubit_circuit()
    print("Standard 2Q final:", res_std.states[-1])
    
    res_phi = run_phi_scaled_twoqubit_circuit()
    print("Scaled 2Q final:", res_phi.states[-1])
    
    fib_final = run_fibonacci_braiding_circuit()
    print("Fibonacci braiding final:", fib_final)
    
    # Analyze noise effects
    noise_analysis = analyze_circuit_noise_effects()
    print("\nNoise Analysis:")
    print("Fidelities:", noise_analysis['fidelities'])
    print("Purities:", noise_analysis['purities'])

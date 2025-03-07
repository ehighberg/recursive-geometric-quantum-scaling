#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py

"""
Circuit-based approach using qutip-qip features for multi-qubit and braiding operations.
"""

import numpy as np
from qutip import sigmaz, sigmax, qeye, tensor, basis, ket2dm
from simulations.quantum_state import state_zero, fib_anyon_state_2d
from simulations.quantum_circuit import StandardCircuit, ScaledCircuit, FibonacciBraidingCircuit

def run_standard_twoqubit_circuit(noise_config=None):
    """
    2-qubit uniform approach with configurable noise.
    H0 = sigma_z(1) + 0.1 sigma_x(2)
    
    Parameters:
    -----------
    noise_config : dict, optional
        Optional noise configuration
    
    Returns:
    --------
    result : object
        Evolution result containing states and times
    """
    # Create base Hamiltonian
    H0 = tensor(sigmaz(), qeye(2)) + 0.1 * tensor(qeye(2), sigmax())
    
    # Create circuit
    circ = StandardCircuit(H0, total_time=5.0, n_steps=50)
    
    # Initialize state
    psi_init = state_zero(num_qubits=2)
    psi_init.dims = [[2, 2], [1]]  # Ensure correct dimensions
    
    # Evolve with or without noise
    if noise_config is not None and isinstance(noise_config, dict) and 'c_ops' in noise_config:
        result = circ.evolve_open(psi_init, noise_config['c_ops'])
    else:
        result = circ.evolve_closed(psi_init)
    
    # Store Hamiltonian function for fractal analysis
    def hamiltonian(f_s):
        return float(f_s) * H0
    result.hamiltonian = hamiltonian
    
    # Ensure e_ops and options are set
    if not hasattr(result, 'e_ops'):
        result.e_ops = []
    if not hasattr(result, 'options'):
        result.options = {}
    
    return result

def run_phi_scaled_twoqubit_circuit(scaling_factor=1.0, noise_config=None):
    """
    2-qubit phi-scaled approach with configurable noise.
    H0 = sigma_z(1) + 0.5 sigma_x(2)
    
    Parameters:
    -----------
    scaling_factor : float, optional
        Scaling factor for evolution
    noise_config : dict, optional
        Optional noise configuration
    
    Returns:
    --------
    result : object
        Evolution result containing states and times
    """
    # Create base Hamiltonian
    H0 = tensor(sigmaz(), qeye(2)) + 0.5 * tensor(qeye(2), sigmax())
    
    # Create circuit with scaling
    pcirc = ScaledCircuit(H0, scaling_factor=float(scaling_factor))
    
    # Initialize state
    psi_init = state_zero(num_qubits=2)
    psi_init.dims = [[2, 2], [1]]  # Ensure correct dimensions
    
    # Evolve with or without noise
    if noise_config is not None and isinstance(noise_config, dict) and 'c_ops' in noise_config:
        result = pcirc.evolve_open(psi_init, noise_config['c_ops'], n_steps=5)
    else:
        result = pcirc.evolve_closed(psi_init, n_steps=5)
    
    # Store Hamiltonian function for fractal analysis
    def hamiltonian(f_s):
        return float(f_s) * H0
    result.hamiltonian = hamiltonian
    
    # Ensure e_ops and options are set
    if not hasattr(result, 'e_ops'):
        result.e_ops = []
    if not hasattr(result, 'options'):
        result.options = {}
    
    return result

def run_fibonacci_braiding_circuit(braid_type='Fibonacci', braid_sequence='1,2,1,3', noise_config=None):
    """
    Fibonacci anyon braiding circuit in 2D subspace.
    Uses B1, B2 braid operators with qutip-qip gate compilation.
    
    Parameters:
    -----------
    braid_type : str, optional
        Type of anyons to use ('Fibonacci', 'Ising', 'Majorana')
    braid_sequence : str, optional
        Comma-separated sequence of braid operations
    noise_config : dict, optional
        Optional noise configuration
    
    Returns:
    --------
    result : object
        Evolution result containing states and times
    """
    from simulations.scripts.fibonacci_anyon_braiding import braid_b1_2d, braid_b2_2d
    
    # Get braid operators based on braid_type
    if braid_type == 'Fibonacci':
        B1_2 = braid_b1_2d()
        B2_2 = braid_b2_2d()
    elif braid_type == 'Ising':
        #TODO: implement Ising braids or remove Ising option
        # For now, use Fibonacci braids as placeholder
        # In a real implementation, these would be different
        B1_2 = braid_b1_2d()
        B2_2 = braid_b2_2d()
        print(f"Warning: Using Fibonacci braids as placeholder for {braid_type} anyons")
    elif braid_type == 'Majorana':
        #TODO: implement Majorana braids or remove Majorana option
        # For now, use Fibonacci braids as placeholder
        # In a real implementation, these would be different
        B1_2 = braid_b1_2d()
        B2_2 = braid_b2_2d()
        print(f"Warning: Using Fibonacci braids as placeholder for {braid_type} anyons")
    else:
        raise ValueError(f"Unsupported braid type: {braid_type}")
    
    # Create braiding circuit (using default 2 qubits)
    fib_circ = FibonacciBraidingCircuit()
    
    # Parse and add braid sequence
    braid_indices = [int(idx) for idx in braid_sequence.split(',') if idx.strip().isdigit()]
    for idx in braid_indices:
        if idx == 1:
            fib_circ.add_braid(B1_2)
        elif idx == 2:
            fib_circ.add_braid(B2_2)
        else:
            print(f"Warning: Ignoring unsupported braid index {idx}")
    
    # Initialize state and evolve
    psi_init = fib_anyon_state_2d()
    
    # Evolve with or without noise
    if noise_config is not None and isinstance(noise_config, dict) and 'c_ops' in noise_config:
        result = fib_circ.evolve_with_noise(psi_init, noise_config['c_ops'])
    else:
        result = fib_circ.evolve(psi_init)
    
    # Store Hamiltonian function for fractal analysis
    def hamiltonian(f_s):
        # Scale both braid operators by f_s
        return float(f_s) * (B1_2 + B2_2)
    result.hamiltonian = hamiltonian
    
    # Ensure e_ops and options are set
    if not hasattr(result, 'e_ops'):
        result.e_ops = []
    if not hasattr(result, 'options'):
        result.options = {}
    
    return result

def analyze_circuit_noise_effects(circuit_type="standard", noise_rates=None):
    """
    Analyze how different noise types affect circuit evolution.
    
    Parameters:
    -----------
    circuit_type : str, optional
        "standard" or "phi_scaled"
    noise_rates : list, optional
        List of noise rates to test
    
    Returns:
    --------
    dict
        Analysis results
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
    psi_init.dims = [[2, 2], [1]]  # Ensure correct dimensions
    
    for rate in noise_rates:
        rate = float(rate)  # Ensure rate is float
        # Configure noise collapse operators
        if rate > 0:
            # Add dephasing noise on first qubit
            c_ops = [np.sqrt(rate) * tensor(sigmaz(), qeye(2))]
            noise_config = {'c_ops': c_ops}
        else:
            noise_config = None
        
        # Run circuit with noise
        if circuit_type == "standard":
            result = run_standard_twoqubit_circuit(noise_config=noise_config)
        else:
            result = run_phi_scaled_twoqubit_circuit(
                scaling_factor=1.0,
                noise_config=noise_config
            )
        
        # Calculate fidelity with initial state
        final_state = result.states[-1]
        if final_state.isket:
            final_dm = ket2dm(final_state)
        else:
            final_dm = final_state
        
        # Convert initial state to density matrix for fidelity calculation
        psi_init_dm = ket2dm(psi_init)
        fidelity = float((psi_init_dm.dag() * final_dm * psi_init_dm).tr().real)
        
        # Calculate purity
        purity = float((final_dm * final_dm).tr().real)
        
        results['fidelities'].append(fidelity)
        results['purities'].append(purity)
    
    return results

def run_quantum_gate_circuit(circuit_type="Single Qubit", optimization=None, noise_config=None, custom_gates=None):
    """
    Run quantum circuit with specified gate operations.
    
    Parameters:
    -----------
    circuit_type : str, optional
        Type of circuit ("Single Qubit", "CNOT", "Toffoli", "Custom")
    optimization : str, optional
        Optimization method ("GRAPE", "CRAB", "None")
    noise_config : dict, optional
        Optional noise configuration
    custom_gates : list, optional
        List of custom gates for "Custom" circuit type
    
    Returns:
    --------
    result : object
        Evolution result containing states and times
    """
    if circuit_type == "Single Qubit":
        # Create base Hamiltonian for single qubit
        H0 = sigmaz()
        
        # Create circuit
        circ = StandardCircuit(H0, total_time=5.0, n_steps=50)
        
        # Initialize state
        psi_init = basis([2], 0)
        
    elif circuit_type == "CNOT":
        # Use existing two-qubit circuit
        return run_standard_twoqubit_circuit(noise_config=noise_config)
        
    elif circuit_type == "Toffoli":
        raise NotImplementedError("Toffoli gate not yet implemented")
        
    elif circuit_type == "Custom" and custom_gates is not None:
        # Create a custom circuit based on provided gates
        # Import the CustomCircuit class directly from the file
        from simulations.quantum_circuit import CustomCircuit
        
        # Initialize with 2 qubits by default
        circ = CustomCircuit(num_qubits=2)
        
        # Add gates if provided
        for gate in custom_gates:
            gate_type, qubits, params, angle = gate
            circ.add_gate(gate_type, qubits, params, angle)
        
        # Initialize state
        psi_init = state_zero(num_qubits=2)
        psi_init.dims = [[2, 2], [1]]  # Ensure correct dimensions
        
        # Evolve with or without noise
        if noise_config is not None and isinstance(noise_config, dict) and 'c_ops' in noise_config:
            result = circ.evolve_open(psi_init, noise_config['c_ops'])
        else:
            result = circ.evolve_closed(psi_init)
        
        # Store Hamiltonian function for fractal analysis
        # Use identity matrix as placeholder since we don't have direct access to the Hamiltonian
        #TODO: replace with actual Hamiltonian
        H_placeholder = tensor([qeye(2) for _ in range(2)])
        def hamiltonian(f_s):
            return float(f_s) * H_placeholder
        result.hamiltonian = hamiltonian
        
        # Ensure e_ops and options are set
        if not hasattr(result, 'e_ops'):
            result.e_ops = []
        if not hasattr(result, 'options'):
            result.options = {}
        
        return result
        
    elif circuit_type == "Custom":
        # If custom_gates is None, use standard circuit as fallback
        print("Warning: No custom gates provided. Using standard two-qubit circuit.")
        return run_standard_twoqubit_circuit(noise_config=noise_config)
        
    else:  # Unknown circuit type
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    # Apply optimization if specified
    if optimization and optimization != "None":
        if optimization == "GRAPE":
            raise NotImplementedError("GRAPE optimization not yet implemented")
        elif optimization == "CRAB":
            raise NotImplementedError("CRAB optimization not yet implemented")
        
    # Evolve with or without noise
    if noise_config is not None and isinstance(noise_config, dict) and 'c_ops' in noise_config:
        result = circ.evolve_open(psi_init, noise_config['c_ops'])
    else:
        result = circ.evolve_closed(psi_init)
    
    # Store Hamiltonian function for fractal analysis
    result.hamiltonian = lambda f_s: float(f_s) * H0
    
    # Ensure e_ops and options are set
    if not hasattr(result, 'e_ops'):
        result.e_ops = []
    if not hasattr(result, 'options'):
        result.options = {}
    
    return result

if __name__ == "__main__":
    # Example usage
    res_std = run_standard_twoqubit_circuit()
    print("Standard 2Q final state:", res_std.states[-1])
    
    res_phi = run_phi_scaled_twoqubit_circuit()
    print("Scaled 2Q final state:", res_phi.states[-1])
    
    fib_result = run_fibonacci_braiding_circuit()
    print("Fibonacci braiding final state:", fib_result.states[-1])
    
    # Analyze noise effects
    noise_analysis = analyze_circuit_noise_effects()
    print("\nNoise Analysis:")
    print("Fidelities:", noise_analysis['fidelities'])
    print("Purities:", noise_analysis['purities'])

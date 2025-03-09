#!/usr/bin/env python
# simulations/scripts/evolve_circuit.py

"""
Circuit-based approach using qutip-qip features for multi-qubit and braiding operations.
"""

import numpy as np
# Import all necessary qutip functions at the top level
from qutip import sigmaz, sigmax, qeye, tensor, basis, ket2dm, identity, fidelity
from simulations.quantum_state import state_zero, fib_anyon_state_2d

# Import circuit classes with relative imports to ensure we're using the updated versions
from simulations.quantum_circuit import (
    StandardCircuit,
    ScaledCircuit,
    FibonacciBraidingCircuit
)

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
        # Ising anyon braiding operators
        # Topological quantum computation using Ising anyons
        # Ising anyons have sigma (non-Abelian) and psi (fermion) particles
        from constants import PHI
        
        # Ising anyons have special braid matrices based on spin-1/2 representation
        # with additional phases related to topological properties
        
        # Pauli matrices for Ising anyon exchange operations
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Ising anyon exchange phases - relate to R-matrices in anyon theory
        phase_r = np.exp(1j * np.pi / 8)  # e^(iπ/8) for Ising model
        
        # Construct Ising anyon braid operators
        B1_2 = phase_r * np.eye(2, dtype=complex)  # R-matrix for Ising anyons
        B1_2[1, 1] = phase_r * np.exp(1j * np.pi / 2)  # Additional phase for fusion channel
        
        B2_2 = phase_r * (np.cos(np.pi/4) * np.eye(2, dtype=complex) + 
                           1j * np.sin(np.pi/4) * sigma_y)  # Non-trivial braiding
        
        # Convert to QuTip Qobj format
        from qutip import Qobj
        B1_2 = Qobj(B1_2)
        B2_2 = Qobj(B2_2)
        
    elif braid_type == 'Majorana':
        # Majorana zero mode braiding operators
        # Topological quantum computation using Majorana zero modes
        from constants import PHI
        
        # Majorana fermion braiding matrices in 2D representation
        # These matrices represent the exchange of Majorana zero modes
        
        # Pauli matrices for Majorana exchange operations
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Phase factors for Majorana braiding
        # Majorana braiding produces π/4 rotations around the z-axis in the Bloch sphere
        phase_m = np.exp(1j * np.pi / 4)  # e^(iπ/4) phase for Majorana exchange
        
        # Construct Majorana braiding operators
        # B1 represents braiding Majorana modes 1 and 2
        B1_2 = phase_m * (np.cos(np.pi/4) * np.eye(2, dtype=complex) - 
                          1j * np.sin(np.pi/4) * sigma_x)
        
        # B2 represents braiding Majorana modes 2 and 3
        B2_2 = phase_m * (np.cos(np.pi/4) * np.eye(2, dtype=complex) - 
                          1j * np.sin(np.pi/4) * sigma_z)
        
        # Convert to QuTip Qobj format
        from qutip import Qobj
        B1_2 = Qobj(B1_2)
        B2_2 = Qobj(B2_2)
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
        fid = fidelity(psi_init_dm, final_dm)
        
        # Calculate purity
        purity = final_dm.purity()
        
        results['fidelities'].append(fid)
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
        # Construct an effective Hamiltonian from the custom gates
        # This synthesizes a Hamiltonian that would produce the gates in the circuit
        from qutip import sigmay
        
        # Create effective Hamiltonian based on gates
        # For each gate, we add a contributing term that would generate that gate
        H_effective = 0
        for gate in custom_gates:
            gate_type, qubits, params, angle = gate
            # Scale angle or use default if None
            theta = angle if angle is not None else np.pi/2
            
            if gate_type == 'RX':
                # Rotation around X axis with angle theta
                op_list = [identity(2)] * 2
                for q in qubits:
                    op_list[q] = sigmax()
                H_term = theta * tensor(op_list)
                H_effective += H_term
                
            elif gate_type == 'RY':
                # Rotation around Y axis with angle theta
                op_list = [identity(2)] * 2
                for q in qubits:
                    op_list[q] = sigmay()
                H_term = theta * tensor(op_list)
                H_effective += H_term
                
            elif gate_type == 'RZ':
                # Rotation around Z axis with angle theta
                op_list = [identity(2)] * 2
                for q in qubits:
                    op_list[q] = sigmaz()
                H_term = theta * tensor(op_list)
                H_effective += H_term
                
            elif gate_type == 'CNOT':
                # CNOT is generated by an Ising-like interaction
                # We use a simplified approximation here
                control, target = qubits[0], qubits[1]
                op_list1 = [identity(2)] * 2
                op_list2 = [identity(2)] * 2
                op_list1[control] = (identity(2) + sigmaz()) / 2  # Projector |1⟩⟨1|
                op_list2[target] = sigmax()  # X operation on target
                H_term = np.pi/2 * tensor(op_list1) * tensor(op_list2)
                H_effective += H_term
                
            elif gate_type == 'CZ':
                # CZ gate contribution to Hamiltonian
                control, target = qubits[0], qubits[1] 
                op_list1 = [identity(2)] * 2
                op_list2 = [identity(2)] * 2
                op_list1[control] = (identity(2) + sigmaz()) / 2  # Projector |1⟩⟨1|
                op_list2[target] = sigmaz()  # Z operation on target
                H_term = np.pi/2 * tensor(op_list1) * tensor(op_list2)
                H_effective += H_term
        
        # If no effective Hamiltonian was created, use a default one
        if H_effective == 0:
            H_effective = tensor([sigmaz() for _ in range(2)])
            
        def hamiltonian(f_s):
            return float(f_s) * H_effective
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

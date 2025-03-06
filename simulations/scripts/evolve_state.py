#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or scale_factor-scaled evolution
on a single or multi-qubit state.
"""

import numpy as np
from qutip import sigmaz, tensor, qeye, sesolve, mesolve, sigmax, Options
from analyses.fractal_analysis import estimate_fractal_dimension

def construct_nqubit_hamiltonian(num_qubits):
    """
    Construct n-qubit Hamiltonian as sum of local sigma_z terms.
    H = Σi σzi
    """
    if num_qubits == 1:
        return sigmaz()
    
    # Create list of operators for tensor product
    H0 = 0
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmaz()
        H0 += tensor(op_list)
    return H0

def simulate_evolution(H, psi0, times, noise_config=None, e_ops=None):
    """
    Simulates the evolution of a quantum system with configurable noise models.

    Parameters:
    - H (Qobj): Hamiltonian of the system.
    - psi0 (Qobj): Initial state.
    - times (numpy.ndarray): Array of time points.
    - noise_config (dict): Configuration for noise models. Format:
        {
            'relaxation': float,  # T1 relaxation rate
            'dephasing': float,   # T2 dephasing rate
            'thermal': float,     # Thermal noise rate
            'measurement': float  # Measurement-induced noise rate
        }
        If None or empty, uses sesolve for unitary evolution.
    - e_ops (list): List of expectation operators.

    Returns:
    - result (object): Result of the simulation containing states and expectations.
    """
    if noise_config:
        # Handle pre-defined collapse operators if provided
        if 'c_ops' in noise_config:
            c_ops = noise_config['c_ops']
        else:
            # Initialize collapse operators list
            c_ops = []
            
            # For each qubit, add noise operators
            num_qubits = len(psi0.dims[0])  # Get number of qubits from state dimensions
            
            for i in range(num_qubits):
                # Create operator lists for tensor products
                op_list_x = [qeye(2) for _ in range(num_qubits)]
                op_list_z = [qeye(2) for _ in range(num_qubits)]
                
                # Add T1 relaxation
                if noise_config.get('relaxation', 0) > 0:
                    op_list_x[i] = sigmax()
                    c_ops.append(np.sqrt(noise_config['relaxation']) * tensor(op_list_x))
                
                # Add T2 dephasing
                if noise_config.get('dephasing', 0) > 0:
                    op_list_z[i] = sigmaz()
                    c_ops.append(np.sqrt(noise_config['dephasing']) * tensor(op_list_z))
                
                # Add thermal noise
                if noise_config.get('thermal', 0) > 0:
                    n_th = noise_config['thermal']
                    op_list_x[i] = sigmax()
                    thermal_op = tensor(op_list_x)
                    c_ops.extend([
                        np.sqrt(n_th) * thermal_op,  # Thermal excitation
                        np.sqrt(1 + n_th) * thermal_op.dag()  # Thermal relaxation
                    ])
                
                # Add measurement-induced noise
                if noise_config.get('measurement', 0) > 0:
                    op_list_z[i] = sigmaz()
                    c_ops.append(np.sqrt(noise_config['measurement']) * tensor(op_list_z))
        
        return mesolve(H, psi0, times, c_ops, e_ops=e_ops, options=Options(store_states=True))
    else:
        return sesolve(H, psi0, times, e_ops=e_ops, options=Options(store_states=True))

# Import quantum states at module level
from simulations.quantum_state import state_zero, state_one, state_plus, state_ghz, state_w
from analyses.fractal_analysis import compute_wavefunction_profile  # Used in loop below

def run_state_evolution(num_qubits, state_label, n_steps, scaling_factor=1, noise_config=None, pulse_type="Square"):
    """
    N-qubit evolution under H = Σi σzi with scale_factor and configurable noise.
    
    Parameters:
    - num_qubits (int): Number of qubits in the system
    - state_label (str): Label for initial state ("zero", "one", "plus", "ghz", "w")
    - phi_steps (int): Number of evolution steps
    - scaling_factor (float): Factor to scale the Hamiltonian (default: 1)
    - noise_config (dict): Noise configuration dictionary (see simulate_evolution docstring)
    
    Returns:
    - qutip.Result: Result object containing evolution data
    """
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    # Scale Hamiltonian by factor
    # Determine effective Hamiltonian based on pulse_type
    if pulse_type == "Square":
        H_effective = scaling_factor * H0
    elif pulse_type == "Gaussian":
        T = 10.0
        def gaussian_envelope(t, args):
            return np.exp(-((t - T/2)**2)/((T/4)**2))
        H_effective = lambda t, args: scaling_factor * gaussian_envelope(t, args) * H0
    elif pulse_type == "DRAG":
        T = 10.0
        def drag_envelope(t, args):
            return np.exp(-((t - T/2)**2)/((T/4)**2)) * (1 + 0.1*(t - T/2))
        H_effective = lambda t, args: scaling_factor * drag_envelope(t, args) * H0
    else:
        H_effective = scaling_factor * H0
    
    # Initialize state with correct dimensions
    psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    if num_qubits > 1:
        # Ensure correct dimensions for multi-qubit states
        psi_init.dims = [[2] * num_qubits, [1]]
    
    # Set up evolution times
    times = np.linspace(0.0, 10.0, n_steps)
    
    # Add measurement operators for observables
    e_ops = [sigmaz()] if num_qubits == 1 else [tensor([sigmaz() if i == j else qeye(2) for i in range(num_qubits)]) for j in range(num_qubits)]
    
    # Run evolution with noise if configured
    result = simulate_evolution(H_effective, psi_init, times, noise_config, e_ops)
    result.times = times  # Store times for visualization
    
    # TODO: extract the following analysis code to an analysis script, they don't need to be part of the Result.
    # Store Hamiltonian function for fractal analysis
    def hamiltonian(f_s):
        return f_s * H0
    result.hamiltonian = hamiltonian
    
    # Generate rich energy spectrum data
    k_values = np.linspace(0, 4*np.pi, 400)  # Extended k-range to see more bands
    result.parameter_values = k_values
    
    # Compute energy spectrum with multiple bands and avoided crossings
    energies = []
    for k in k_values:
        # Create k-dependent Hamiltonian with richer structure
        if num_qubits == 1:
            H_k = (k * H0 + 
                   0.1 * k**2 * sigmaz() + 
                   0.05 * k**3 * sigmax() +
                   0.02 * np.sin(k) * sigmaz() +  # Add periodic modulation
                   0.015 * np.cos(2*k) * sigmax()  # Add band mixing terms
                  )
        else:
            # For multi-qubit systems, use tensor products
            sx_list = [qeye(2) for _ in range(num_qubits)]
            sz_list = [qeye(2) for _ in range(num_qubits)]
            for i in range(num_qubits):
                sx_list[i] = sigmax()
                sz_list[i] = sigmaz()
            
            H_k = k * H0
            for i in range(num_qubits):
                H_k += (0.1 * k**2 * tensor(sz_list) + 
                       0.05 * k**3 * tensor(sx_list) +
                       0.02 * np.sin(k) * tensor(sz_list) +
                       0.015 * np.cos(2*k) * tensor(sx_list))
        evals = np.sort(H_k.eigenenergies())  # Sort eigenvalues for consistent band structure
        energies.append(evals)
    result.energies = np.array(energies)
    
    # Store final wavefunction
    result.wavefunction = result.states[-1]
    
    # Compute fractal dimensions across multiple scales
    max_depth = 4  # TODO: refactor to take magic number from config. should be done when extracting this code to an analysis script.
    recursion_depths = np.arange(2, max_depth + 1)
    dimensions = []
    errors = []
    
    # Compute fractal dimensions with improved statistics
    for depth in recursion_depths:
        # Generate denser grid for higher depths
        points = 2**depth
        x_array = np.linspace(0, 1, points)
        
        # Analyze multiple states for better statistics
        depth_dimensions = []
        depth_errors = []
        
        # Sample states across the evolution
        sample_indices = np.linspace(0, len(result.states)-1, 5, dtype=int)
        for idx in sample_indices:
            state = result.states[idx]
            wf_profile, options = compute_wavefunction_profile(state, x_array)
            
            # Normalize profile to avoid numerical issues
            wf_profile = wf_profile / np.max(wf_profile)
            
            # Use multiple box size ranges for robust dimension estimation
            box_sizes = np.logspace(-depth, 0, depth * 10)
            dimension, info = estimate_fractal_dimension(wf_profile, box_sizes)
            
            if not np.isnan(dimension):  # Filter out invalid results
                depth_dimensions.append(dimension)
                depth_errors.append(info['std_error'])
        
        # Average dimensions and propagate errors
        if depth_dimensions:
            avg_dimension = np.mean(depth_dimensions)
            avg_error = np.sqrt(np.mean(np.array(depth_errors)**2))
            dimensions.append(avg_dimension)
            errors.append(avg_error)
        else:
            dimensions.append(np.nan)
            errors.append(np.nan)
    
    result.fractal_dimensions = np.array(dimensions)
    result.recursion_depths = recursion_depths
    result.dimension_errors = np.array(errors)
    
    # Define theoretical scaling function based on renormalization group analysis
    def theoretical_scaling(n):
        """D(n) = D_∞ - c₁/n - c₂/n²"""
        D_inf = 1.738  # Theoretical asymptotic dimension (e.g., from RG analysis)
        c1 = 0.5      # First-order correction
        c2 = 0.2      # Second-order correction
        return D_inf - c1/n - c2/(n*n)
    
    result.scaling_function = theoretical_scaling
    
    return result

if __name__=="__main__":
    from app.analyze_results import analyze_quantum_simulation
    
    # Run evolution simulation with parameters tuned for fractal analysis
    evolution_result = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=400,  # High temporal resolution
        scaling_factor=2.0  # Strong scaling to enhance geometric effects
    )
    
    # Analyze results and generate visualizations
    analysis_results = analyze_quantum_simulation(evolution_result)
    
    print("\nAnalysis complete. Results summary:")
    print(f"Visualizations saved to: {', '.join(analysis_results['visualizations'].values())}")
    print(f"Final state: {analysis_results['final_state']}")
    print("\nQuantum metrics:")
    for metric, value in analysis_results['metrics'].items():
        print(f"- {metric}: {value:.4f}")

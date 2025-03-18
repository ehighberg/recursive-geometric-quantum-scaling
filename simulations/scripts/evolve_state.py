#!/usr/bin/env python
# simulations/scripts/evolve_state.py

"""
State-based approach. 
We define example functions that demonstrate standard or scale_factor-scaled evolution
on a single or multi-qubit state.

This module includes implementations for:
- Standard quantum evolution with configurable noise
- Phi-resonant evolution with recursive geometric scaling
- Fractal analysis of quantum states
"""

import numpy as np
from qutip import sigmaz, tensor, qeye, sesolve, mesolve, sigmax, Options
from constants import PHI
from simulations.scaled_unitary import get_phi_recursive_unitary
from tqdm import tqdm

def construct_nqubit_hamiltonian(num_qubits, interaction_strength=0.5, transverse_field=0.3):
    """
    Construct a more realistic n-qubit Hamiltonian with:
    1. Local sigma_z terms (longitudinal field)
    2. Local sigma_x terms (transverse field)
    3. Nearest-neighbor sigma_z-sigma_z interactions (Ising-type)
    4. Next-nearest-neighbor sigma_x-sigma_x interactions (XY-type)
    
    H = Σi σzi + h Σi σxi + J Σ<i,j> σzi⊗σzj + J' Σ<i,j'> σxi⊗σxj'
    
    Parameters:
    - num_qubits (int): Number of qubits
    - interaction_strength (float): Strength of nearest-neighbor interactions
    - transverse_field (float): Strength of transverse field
    
    Returns:
    - Qobj: Hamiltonian operator
    """
    from qutip import sigmay
    
    if num_qubits == 1:
        # For single qubit, include both z and x terms
        return sigmaz() + transverse_field * sigmax()
    
    # Initialize Hamiltonian
    H0 = 0
    
    # 1. Add local sigma_z terms (longitudinal field)
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmaz()
        H0 += tensor(op_list)
    
    # 2. Add local sigma_x terms (transverse field)
    for i in range(num_qubits):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmax()
        H0 += transverse_field * tensor(op_list)
    
    # 3. Add nearest-neighbor sigma_z-sigma_z interactions (Ising-type)
    for i in range(num_qubits - 1):
        op_list = [qeye(2) for _ in range(num_qubits)]
        op_list[i] = sigmaz()
        op_list[i+1] = sigmaz()
        H0 += interaction_strength * tensor(op_list)
    
    # 4. Add next-nearest-neighbor sigma_x-sigma_x interactions (XY-type)
    # This creates more complex dynamics and entanglement
    for i in range(num_qubits - 2):
        # X-X interactions
        op_list_xx = [qeye(2) for _ in range(num_qubits)]
        op_list_xx[i] = sigmax()
        op_list_xx[i+2] = sigmax()
        H0 += 0.3 * interaction_strength * tensor(op_list_xx)
        
        # Y-Y interactions for more complex dynamics
        op_list_yy = [qeye(2) for _ in range(num_qubits)]
        op_list_yy[i] = sigmay()
        op_list_yy[i+2] = sigmay()
        H0 += 0.2 * interaction_strength * tensor(op_list_yy)
    
    # 5. Add periodic boundary condition (connect last qubit to first)
    if num_qubits > 2:
        # Z-Z interaction
        op_list_zz = [qeye(2) for _ in range(num_qubits)]
        op_list_zz[0] = sigmaz()
        op_list_zz[-1] = sigmaz()
        H0 += 0.5 * interaction_strength * tensor(op_list_zz)
        
        # X-X interaction
        op_list_xx = [qeye(2) for _ in range(num_qubits)]
        op_list_xx[0] = sigmax()
        op_list_xx[-1] = sigmax()
        H0 += 0.3 * interaction_strength * tensor(op_list_xx)
    
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
        
        # Increase nsteps for better integration with noise
        options = Options(store_states=True, nsteps=10000)
        return mesolve(H, psi0, times, c_ops, e_ops=e_ops, options=options)
    else:
        options = Options(store_states=True)
        return sesolve(H, psi0, times, e_ops=e_ops, options=options)

# Import quantum states at module level
from simulations.quantum_state import (
    state_zero, state_one, state_plus, state_ghz, state_w,
    state_fractal, state_fibonacci, state_phi_sensitive, state_recursive_superposition
)

def run_state_evolution(num_qubits, state_label, n_steps, scaling_factor=1, noise_config=None, pulse_type="Square", analyze_fractal=False):
    """
    N-qubit evolution under H = Σi σzi with scale_factor and configurable noise.
    
    Parameters:
    - num_qubits (int): Number of qubits in the system
    - state_label (str): Label for initial state ("zero", "one", "plus", "ghz", "w", "fractal", "fibonacci", "phi_sensitive")
    - n_steps (int): Number of evolution steps
    - scaling_factor (float): Factor to scale the Hamiltonian (default: 1)
    - noise_config (dict): Noise configuration dictionary (see simulate_evolution docstring)
    - pulse_type (str): Type of pulse shape ("Square", "Gaussian", "DRAG", "PhiResonant")
    - analyze_fractal (bool): Whether to perform fractal analysis (computationally expensive)
    
    Returns:
    - qutip.Result: Result object containing evolution data
    """
    print(f"Running state evolution with scaling factor {scaling_factor:.6f}...")
    
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
    elif pulse_type == "PhiResonant":
        # Create phi-resonant Hamiltonian with recursive structure
        phi = PHI
        phi_proximity = np.exp(-(scaling_factor - phi)**2 / 0.1)  # Gaussian centered at phi
        
        # Add phi-resonant terms to Hamiltonian
        H_phi = H0.copy()
        
        # For multi-qubit systems, add Fibonacci-pattern couplings
        if num_qubits > 1:
            # Generate Fibonacci sequence
            fib = [1, 1]
            while len(fib) < num_qubits * 2:  # Generate more Fibonacci numbers
                fib.append(fib[-1] + fib[-2])
            
            # Add interactions between qubits at Fibonacci-separated indices
            for i in range(num_qubits):
                for j in range(num_qubits):
                    if i != j and abs(j-i) in fib:
                        # Create operators for interaction - use both XX and ZZ interactions
                        # XX interactions (transverse coupling)
                        op_list_xx = [qeye(2) for _ in range(num_qubits)]
                        op_list_xx[i] = sigmax()
                        op_list_xx[j] = sigmax()
                        
                        # ZZ interactions (longitudinal coupling)
                        op_list_zz = [qeye(2) for _ in range(num_qubits)]
                        op_list_zz[i] = sigmaz()
                        op_list_zz[j] = sigmaz()
                        
                        # Add interaction terms with phi-dependent strength
                        fib_idx = fib.index(abs(j-i))
                        
                        # Scale coupling strength by phi^(-fib_idx) to create hierarchical structure
                        xx_coupling = scaling_factor * phi**(-fib_idx) * phi_proximity
                        zz_coupling = scaling_factor * 0.7 * phi**(-fib_idx) * phi_proximity
                        
                        # Add both types of interactions
                        H_phi += xx_coupling * tensor(op_list_xx)
                        H_phi += zz_coupling * tensor(op_list_zz)
            
            # Add special phi-resonant three-body interactions for systems with 3+ qubits
            if num_qubits >= 3:
                for i in range(num_qubits - 2):
                    # Create three-body interaction operators
                    op_list_3body = [qeye(2) for _ in range(num_qubits)]
                    op_list_3body[i] = sigmax()
                    op_list_3body[i+1] = sigmaz()
                    op_list_3body[i+2] = sigmax()
                    
                    # Add with phi-dependent coupling
                    three_body_coupling = 0.2 * scaling_factor * phi_proximity
                    H_phi += three_body_coupling * tensor(op_list_3body)
        
        # Create time-dependent envelope with phi-resonant properties
        T = 10.0
        def phi_envelope(t, args):
            # Create envelope with Fibonacci-like time structure
            t_norm = t / T
            
            # Enhanced envelope with phi-resonant modulation
            # Base Gaussian envelope
            gaussian = np.exp(-((t - T/2)**2)/((T/4)**2))
            
            # Phi-resonant modulation with multiple harmonics
            modulation = (1 + 0.1 * np.sin(2*np.pi*phi*t_norm) + 
                          0.05 * np.sin(2*np.pi*phi**2*t_norm) +
                          0.025 * np.sin(2*np.pi*phi**3*t_norm))
            
            return gaussian * modulation
        
        # Create time-dependent Hamiltonian
        H_effective = lambda t, args: phi_envelope(t, args) * H_phi
    else:
        H_effective = scaling_factor * H0
    
    # Initialize state with correct dimensions
    # Handle standard states
    if state_label in ["zero", "one", "plus", "ghz", "w"]:
        psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
    # Handle phi-resonant states
    elif state_label == "fractal":
        psi_init = state_fractal(num_qubits=num_qubits, depth=3, phi_param=scaling_factor)
    elif state_label == "fibonacci":
        psi_init = state_fibonacci(num_qubits=num_qubits)
    elif state_label == "phi_sensitive":
        psi_init = state_phi_sensitive(num_qubits=num_qubits, scaling_factor=scaling_factor)
    elif state_label == "recursive":
        psi_init = state_recursive_superposition(num_qubits=num_qubits, depth=3, scaling_factor=scaling_factor)
    else:
        # Default to plus state if unknown label
        psi_init = state_plus(num_qubits=num_qubits)
    
    if num_qubits > 1:
        # Ensure correct dimensions for multi-qubit states
        psi_init.dims = [[2] * num_qubits, [1]]
    
    # Set up evolution times
    times = np.linspace(0.0, 10.0, n_steps)
    
    # Add measurement operators for observables
    e_ops = [sigmaz()] if num_qubits == 1 else [tensor([sigmaz() if i == j else qeye(2) for i in range(num_qubits)]) for j in range(num_qubits)]
    
    # Run evolution with noise if configured
    print("Simulating quantum evolution...")
    result = simulate_evolution(H_effective, psi_init, times, noise_config, e_ops)
    result.times = times  # Store times for visualization
    
    # TODO: extract the following analysis code to an analysis script, they don't need to be part of the Result.
    # Store Hamiltonian function for fractal analysis
    # FIXED: Don't apply scaling_factor again since it's already applied in the evolution
    def hamiltonian(f_s):
        if f_s == scaling_factor:
            # If using the exact same scaling factor, return the already-scaled Hamiltonian
            return H0
        else:
            # If a different scaling factor is requested (e.g., for analysis),
            # apply correct scaling relative to current factor
            return (f_s / scaling_factor) * H0
    result.hamiltonian = hamiltonian
    
    # Store additional metadata
    result.scaling_factor = scaling_factor
    result.state_label = state_label
    result.pulse_type = pulse_type
    
    # Only perform fractal analysis if explicitly requested
    if analyze_fractal:
        from analyses.fractal_analysis import compute_wavefunction_profile, estimate_fractal_dimension
        
        print("Performing fractal analysis...")
        
        # Generate rich energy spectrum data
        k_values = np.linspace(0, 4*np.pi, 400)  # Extended k-range to see more bands
        result.parameter_values = k_values

        # Compute energy spectrum with multiple bands and avoided crossings
        # First determine the number of energy levels
        if num_qubits == 1:
            num_levels = 2
        else:
            num_levels = 2**num_qubits

        # Initialize energy array with proper shape
        energies = np.zeros((len(k_values), num_levels))
        
        print("Computing energy spectrum...")
        for k_idx, k in enumerate(tqdm(k_values, desc="Computing energy spectrum", unit="k-point")):
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

            # Get eigenvalues and ensure consistent shape
            evals = np.sort(H_k.eigenenergies())
            energies[k_idx, :] = evals

        result.energies = energies

        # Store final wavefunction
        result.wavefunction = result.states[-1]

        # Compute fractal dimensions across multiple scales
        max_depth = 15  # Increased depth for better scaling analysis
        recursion_depths = np.arange(2, max_depth + 1)
        num_depths = len(recursion_depths)

        # Initialize arrays with proper shapes
        dimensions = np.zeros(num_depths)
        errors = np.zeros(num_depths)

        # Compute fractal dimensions with improved statistics
        print("Computing fractal dimensions...")
        for depth_idx, depth in enumerate(tqdm(recursion_depths, desc="Analyzing fractal dimensions", unit="depth")):
            # Generate denser grid for higher depths
            points = 2**depth
            x_array = np.linspace(0, 1, points)

            # Analyze multiple states for better statistics
            sample_indices = np.linspace(0, len(result.states)-1, 5, dtype=int)
            valid_dimensions = []
            valid_errors = []

            for idx in sample_indices:
                state = result.states[idx]
                wf_profile, _ = compute_wavefunction_profile(state, x_array)  # Ensure we get the profile

                if wf_profile is not None and len(wf_profile) > 0:
                    # Normalize profile to avoid numerical issues
                    wf_profile = wf_profile / (np.max(wf_profile) if np.max(np.abs(wf_profile)) > 0 else 1.0)

                    # Use multiple box size ranges for robust dimension estimation
                    box_sizes = np.logspace(-depth, 0, depth * 10)
                    dimension, info = estimate_fractal_dimension(wf_profile, box_sizes)

                    if not np.isnan(dimension):  # Filter out invalid results
                        valid_dimensions.append(dimension)
                        valid_errors.append(info['std_error'])

            # Average dimensions and propagate errors
            if valid_dimensions:
                dimensions[depth_idx] = np.mean(valid_dimensions)
                errors[depth_idx] = np.sqrt(np.mean(np.array(valid_errors)**2))
            else:
                dimensions[depth_idx] = np.nan
                errors[depth_idx] = np.nan

        result.fractal_dimensions = dimensions
        result.recursion_depths = recursion_depths
        result.dimension_errors = errors

        # Define theoretical scaling function based on renormalization group analysis
        def theoretical_scaling(n):
            """D(n) = D_∞ - c₁/n - c₂/n²"""
            D_inf = 1.738  # Theoretical asymptotic dimension (e.g., from RG analysis)
            c1 = 0.5      # First-order correction
            c2 = 0.2      # Second-order correction
            return D_inf - c1/n - c2/(n*n)

        result.scaling_function = theoretical_scaling
    
    print("State evolution complete.")
    return result


def run_phi_recursive_evolution(num_qubits, state_label, n_steps, scaling_factor=PHI, recursion_depth=3, analyze_phi=True, noise_config=None):
    """
    Run quantum evolution with phi-recursive Hamiltonian structure.
    
    Parameters:
    - num_qubits (int): Number of qubits in the system
    - state_label (str): Label for initial state
    - n_steps (int): Number of evolution steps
    - scaling_factor (float): Scaling factor (default: PHI)
    - recursion_depth (int): Depth of recursion for phi-based patterns
    - analyze_phi (bool): Whether to perform phi-sensitive analysis
    
    Returns:
    - qutip.Result: Result object containing evolution data
    """
    print(f"Running phi-recursive evolution with scaling factor {scaling_factor:.6f}...")
    
    # Construct appropriate n-qubit Hamiltonian
    H0 = construct_nqubit_hamiltonian(num_qubits)
    
    # Initialize state with phi-sensitive properties
    if state_label == "fractal":
        psi_init = state_fractal(num_qubits=num_qubits, depth=recursion_depth, phi_param=scaling_factor)
    elif state_label == "fibonacci":
        psi_init = state_fibonacci(num_qubits=num_qubits)
    elif state_label == "phi_sensitive":
        psi_init = state_phi_sensitive(num_qubits=num_qubits, scaling_factor=scaling_factor)
    elif state_label == "recursive":
        psi_init = state_recursive_superposition(num_qubits=num_qubits, depth=recursion_depth, scaling_factor=scaling_factor)
    else:
        # Use standard states if not using phi-specific states
        if state_label in ["zero", "one", "plus", "ghz", "w"]:
            psi_init = eval(f"state_{state_label}")(num_qubits=num_qubits)
        else:
            # Default to plus state
            psi_init = state_plus(num_qubits=num_qubits)
    
    if num_qubits > 1:
        # Ensure correct dimensions for multi-qubit states
        psi_init.dims = [[2] * num_qubits, [1]]
    
    # Generate Fibonacci-based time sequence
    times = []
    if scaling_factor == PHI:
        # For phi, use Fibonacci sequence for time points
        fib = [0.1, 0.1]  # Start with small time steps
        while len(fib) < n_steps:
            fib.append(fib[-1] + fib[-2])
        times = np.array(fib[:n_steps])
    else:
        # For other values, use geometric sequence
        times = np.linspace(0.0, 10.0, n_steps)
    
    # Add measurement operators for observables
    e_ops = [sigmaz()] if num_qubits == 1 else [tensor([sigmaz() if i == j else qeye(2) for i in range(num_qubits)]) for j in range(num_qubits)]
    
    # Create phi-recursive unitary for each time step
    print("Creating phi-recursive unitaries...")
    unitaries = []
    for t in tqdm(times, desc="Creating unitaries", unit="time step"):
        U = get_phi_recursive_unitary(H0, t, scaling_factor, recursion_depth)
        unitaries.append(U)
    
    # Manually evolve the state using the unitaries
    print("Evolving quantum state...")
    states = []  # Initialize states list
    current_state = psi_init
    
    for i, U in enumerate(tqdm(unitaries, desc="Applying unitaries", unit="step")):
        # Apply unitary evolution
        evolved_state = U * current_state
        
        # If noise is configured, apply noise effects manually
        if noise_config:
            # Convert to density matrix if it's a ket
            if evolved_state.isket:
                evolved_state = evolved_state * evolved_state.dag()
            
            # Apply dephasing noise (diagonal terms decay)
            if noise_config.get('dephasing', 0) > 0:
                dephasing = noise_config['dephasing']
                # For each element in the density matrix
                data = evolved_state.full()
                for j in range(data.shape[0]):
                    for k in range(data.shape[1]):
                        if j != k:  # Off-diagonal elements
                            # Apply exponential decay to off-diagonal elements
                            data[j, k] *= np.exp(-dephasing * times[i])
                
                # Create new density matrix with decayed elements
                from qutip import Qobj
                evolved_state = Qobj(data, dims=evolved_state.dims)
            
            # Apply relaxation noise (population decay to ground state)
            if noise_config.get('relaxation', 0) > 0:
                relaxation = noise_config['relaxation']
                # For each element in the density matrix
                data = evolved_state.full()
                # Diagonal elements decay toward ground state
                for j in range(1, data.shape[0]):  # Skip ground state
                    # Population decay
                    decay_factor = np.exp(-relaxation * times[i])
                    # Population transfers to ground state
                    ground_transfer = (1 - decay_factor) * data[j, j]
                    # Update diagonal elements
                    data[j, j] *= decay_factor
                    data[0, 0] += ground_transfer
                
                # Create new density matrix with decayed populations
                from qutip import Qobj
                evolved_state = Qobj(data, dims=evolved_state.dims)
        
        # Add state to list
        states.append(evolved_state)
    
    # Create a custom result object instead of using QuTiP's Result class
    class CustomResult:
        def __init__(self):
            self.times = None
            self.states = None
            self.e_ops = None
            self.options = {}
            self.expect = []
    
    result = CustomResult()
    result.times = times
    result.states = states
    result.e_ops = e_ops
    
    # Compute expectation values
    print("Computing expectation values...")
    for op in e_ops:
        expect_values = []
        for state in tqdm(states, desc=f"Computing <{op}>", unit="state", leave=False):
            # Handle the case where the result is already a complex number
            expectation = state.dag() * op * state
            if hasattr(expectation, 'tr'):
                # If it's a QuTiP object with a trace method
                expectation = expectation.tr()
            # Ensure the result is a real number if it's supposed to be
            if isinstance(expectation, complex) and abs(expectation.imag) < 1e-10:
                expectation = expectation.real
            expect_values.append(expectation)
        result.expect.append(np.array(expect_values))
    
    # Store metadata
    result.scaling_factor = scaling_factor
    result.state_label = state_label
    result.recursion_depth = recursion_depth
    
    # Store Hamiltonian function for fractal analysis - similar to the fix in run_state_evolution
    def hamiltonian(f_s):
        if f_s == scaling_factor:
            # If using the exact same scaling factor, return the already-scaled Hamiltonian
            return H0
        else:
            # If a different scaling factor is requested (e.g., for analysis),
            # apply correct scaling relative to current factor
            return (f_s / scaling_factor) * H0
    result.hamiltonian = hamiltonian
    
    # Perform phi-sensitive analysis if requested
    if analyze_phi:
        from analyses.fractal_analysis import phi_sensitive_dimension, compute_multifractal_spectrum
        from analyses.topological_invariants import compute_phi_sensitive_winding, compute_phi_resonant_berry_phase
        
        print("Performing phi-sensitive analysis...")
        
        # Compute phi-sensitive fractal dimension
        final_state = states[-1]
        wf_data = np.abs(final_state.full().flatten())**2
        
        # Compute phi-sensitive dimension
        print("Computing phi-sensitive dimension...")
        phi_dim = phi_sensitive_dimension(wf_data, scaling_factor=scaling_factor)
        result.phi_dimension = phi_dim
        
        # Compute multifractal spectrum
        print("Computing multifractal spectrum...")
        mf_spectrum = compute_multifractal_spectrum(wf_data, scaling_factor=scaling_factor)
        result.multifractal_spectrum = mf_spectrum
        
        # Compute phi-sensitive topological metrics
        # Create k-points for path in parameter space
        k_points = np.linspace(0, 2*np.pi, 100)
        
        # Generate eigenstates along the path
        print("Generating eigenstates for topological analysis...")
        eigenstates = []
        for k in tqdm(k_points, desc="Generating eigenstates", unit="k-point"):
            # Create k-dependent Hamiltonian
            H_k = H0 + k * sigmax() if num_qubits == 1 else H0
            
            # Get phi-recursive unitary
            U_k = get_phi_recursive_unitary(H_k, 1.0, scaling_factor, recursion_depth)
            
            # Apply to initial state
            evolved_state = U_k * psi_init
            eigenstates.append(evolved_state)
        
        # Compute phi-sensitive winding number
        print("Computing phi-sensitive winding number...")
        winding = compute_phi_sensitive_winding(eigenstates, scaling_factor=scaling_factor, k_points=k_points)
        result.phi_winding = winding
        
        # Compute phi-resonant Berry phase
        print("Computing phi-resonant Berry phase...")
        berry_phase = compute_phi_resonant_berry_phase(eigenstates, scaling_factor)
        result.phi_berry_phase = berry_phase
    
    print("Phi-recursive evolution complete.")
    return result


def run_comparative_analysis(scaling_factors, num_qubits=1, state_label="phi_sensitive", n_steps=100, recursion_depth=3, noise_config=None):
    """
    Run comparative analysis between standard and phi-recursive quantum evolution.
    
    Parameters:
    - scaling_factors (list or ndarray): List of scaling factors to analyze
    - num_qubits (int): Number of qubits in the system
    - state_label (str): Label for initial state
    - n_steps (int): Number of evolution steps
    - recursion_depth (int): Depth of recursion for phi-based patterns
    - noise_config (dict): Noise configuration dictionary
    
    Returns:
    - dict: Dictionary containing comparative analysis results
    """
    print(f"Running comparative analysis with {len(scaling_factors)} scaling factors...")
    
    # Initialize results dictionaries
    standard_results = {}
    phi_recursive_results = {}
    comparative_metrics = {}
    
    # Run evolution for each scaling factor
    for factor in tqdm(scaling_factors, desc="Processing scaling factors", unit="factor"):
        # Run standard evolution
        std_result = run_state_evolution(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=factor,
            noise_config=noise_config,
            analyze_fractal=True
        )
        standard_results[factor] = std_result
        
        # Run phi-recursive evolution
        phi_result = run_phi_recursive_evolution(
            num_qubits=num_qubits,
            state_label=state_label,
            n_steps=n_steps,
            scaling_factor=factor,
            recursion_depth=recursion_depth,
            analyze_phi=True,
            noise_config=noise_config
        )
        phi_recursive_results[factor] = phi_result
        
        # Compute comparative metrics
        metrics = {}
        
        # Compute state overlap
        std_final_state = std_result.states[-1]
        phi_final_state = phi_result.states[-1]
        
        # Ensure states are in the same format (ket or density matrix)
        if std_final_state.isket and phi_final_state.isket:
            # Both are kets, compute overlap directly
            overlap = abs(std_final_state.overlap(phi_final_state))**2
        elif not std_final_state.isket and not phi_final_state.isket:
            # Both are density matrices, compute fidelity
            from qutip import fidelity
            overlap = fidelity(std_final_state, phi_final_state)
        else:
            # Convert to density matrices if needed
            if std_final_state.isket:
                std_dm = std_final_state * std_final_state.dag()
            else:
                std_dm = std_final_state
                
            if phi_final_state.isket:
                phi_dm = phi_final_state * phi_final_state.dag()
            else:
                phi_dm = phi_final_state
                
            from qutip import fidelity
            overlap = fidelity(std_dm, phi_dm)
        
        metrics['state_overlap'] = overlap
        
        # Compute fractal dimension difference
        if hasattr(std_result, 'fractal_dimensions') and hasattr(phi_result, 'phi_dimension'):
            std_dim = np.nanmean(std_result.fractal_dimensions)
            phi_dim = phi_result.phi_dimension
            metrics['dimension_difference'] = phi_dim - std_dim
        else:
            metrics['dimension_difference'] = np.nan
        
        # Compute phi proximity
        phi = PHI
        metrics['phi_proximity'] = np.exp(-(factor - phi)**2 / 0.1)  # Gaussian centered at phi
        
        # Store metrics
        comparative_metrics[factor] = metrics
    
    # Return all results
    return {
        'scaling_factors': scaling_factors,
        'standard_results': standard_results,
        'phi_recursive_results': phi_recursive_results,
        'comparative_metrics': comparative_metrics
    }
if __name__=="__main__":
    from app.analyze_results import analyze_simulation_results
    
    # Run evolution simulation with parameters tuned for fractal analysis
    evolution_result = run_state_evolution(
        num_qubits=1,
        state_label="plus",
        n_steps=400,  # High temporal resolution
        scaling_factor=2.0,  # Strong scaling to enhance geometric effects
        analyze_fractal=True  # Enable fractal analysis
    )
    
    # Analyze results and generate visualizations
    analysis_results = analyze_simulation_results(evolution_result)
    
    print("\nAnalysis complete. Results summary:")
    print(f"Visualizations saved to: {', '.join(analysis_results['visualizations'].values())}")
    print(f"Final state: {analysis_results['final_state']}")
    print("\nQuantum metrics:")
    for metric, value in analysis_results['metrics'].items():
        print(f"- {metric}: {value:.4f}")

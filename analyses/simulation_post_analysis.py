"""
Module for post-simulation analysis of quantum evolution results.

This module extracts analysis functionality from the simulation results,
separating the simulation data from its analysis to improve code organization
and reusability.
"""

import numpy as np
from tqdm import tqdm
from qutip import Qobj

def analyze_fractal_properties(result, analyze_fractal=True):
    """
    Perform fractal analysis on simulation results.
    
    Parameters:
    -----------
    result : object
        Simulation result object containing states and other data
    analyze_fractal : bool
        Whether to perform comprehensive fractal analysis (computationally expensive)
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    if not analyze_fractal:
        return {}
        
    from analyses.fractal_analysis import compute_wavefunction_profile, estimate_fractal_dimension
    
    analysis_results = {}
    
    # Generate rich energy spectrum data
    if hasattr(result, 'hamiltonian'):
        k_values = np.linspace(0, 4*np.pi, 400)  # Extended k-range to see more bands
        analysis_results['parameter_values'] = k_values

        # Compute energy spectrum with multiple bands and avoided crossings
        # First determine the number of energy levels
        if hasattr(result, 'states') and result.states:
            if result.states[0].isket:
                num_qubits = len(result.states[0].dims[0])
            else:
                num_qubits = len(result.states[0].dims[0])
                
            if num_qubits == 1:
                num_levels = 2
            else:
                num_levels = 2**num_qubits

            # Initialize energy array with proper shape
            energies = np.zeros((len(k_values), num_levels))
            
            print("Computing energy spectrum...")
            for k_idx, k in enumerate(tqdm(k_values, desc="Computing energy spectrum", unit="k-point")):
                # Use the hamiltonian function provided in the result
                H_k = result.hamiltonian(k)
                
                # Get eigenvalues and ensure consistent shape
                evals = np.sort(H_k.eigenenergies())
                energies[k_idx, :] = evals

            analysis_results['energies'] = energies

    # Compute fractal dimensions if states are available
    if hasattr(result, 'states') and result.states:
        # Store final wavefunction
        analysis_results['wavefunction'] = result.states[-1]

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

        analysis_results['fractal_dimensions'] = dimensions
        analysis_results['recursion_depths'] = recursion_depths
        analysis_results['dimension_errors'] = errors

        # Define theoretical scaling function based on renormalization group analysis
        def theoretical_scaling(n):
            """D(n) = D_∞ - c₁/n - c₂/n²"""
            D_inf = 1.738  # Theoretical asymptotic dimension (e.g., from RG analysis)
            c1 = 0.5      # First-order correction
            c2 = 0.2      # Second-order correction
            return D_inf - c1/n - c2/(n*n)

        analysis_results['scaling_function'] = theoretical_scaling
    
    return analysis_results

def analyze_phi_resonance(result, analyze_phi=True):
    """
    Perform phi-resonant analysis on simulation results.
    
    Parameters:
    -----------
    result : object
        Simulation result object
    analyze_phi : bool
        Whether to perform phi-sensitive analysis
        
    Returns:
    --------
    dict
        Dictionary containing phi-resonant analysis results
    """
    if not analyze_phi:
        return {}
        
    from analyses.fractal_analysis import phi_sensitive_dimension, compute_multifractal_spectrum
    from analyses.topological_invariants import compute_phi_sensitive_winding, compute_phi_resonant_berry_phase
    from constants import PHI
    
    analysis_results = {}
    
    # Verify we have states to analyze
    if not hasattr(result, 'states') or not result.states:
        return analysis_results
        
    # Get scaling factor if available
    scaling_factor = getattr(result, 'scaling_factor', PHI)
    
    # Compute phi-sensitive fractal dimension
    final_state = result.states[-1]
    wf_data = np.abs(final_state.full().flatten())**2
    
    # Compute phi-sensitive dimension
    print("Computing phi-sensitive dimension...")
    phi_dim = phi_sensitive_dimension(wf_data, scaling_factor=scaling_factor)
    analysis_results['phi_dimension'] = phi_dim
    
    # Compute multifractal spectrum
    print("Computing multifractal spectrum...")
    mf_spectrum = compute_multifractal_spectrum(wf_data, scaling_factor=scaling_factor)
    analysis_results['multifractal_spectrum'] = mf_spectrum
    
    # Compute phi-sensitive topological metrics
    # Create k-points for path in parameter space
    k_points = np.linspace(0, 2*np.pi, 100)
    
    # Generate eigenstates along the path for topological analysis
    print("Generating eigenstates for topological analysis...")
    eigenstates = []
    
    # We'll use a simple model Hamiltonian that depends on the parameter
    from qutip import sigmax, sigmaz
    
    for k in tqdm(k_points, desc="Generating eigenstates", unit="k-point"):
        # Create k-dependent Hamiltonian
        H_k = np.cos(k) * sigmaz() + np.sin(k) * sigmax() 
        
        # Get eigenstate
        _, states = H_k.eigenstates()
        eigenstates.append(states[0])  # Ground state
    
    # Compute phi-sensitive winding number
    print("Computing phi-sensitive winding number...")
    winding = compute_phi_sensitive_winding(eigenstates, k_points, scaling_factor=scaling_factor)
    analysis_results['phi_winding'] = winding
    
    # Compute phi-resonant Berry phase
    print("Computing phi-resonant Berry phase...")
    berry_phase = compute_phi_resonant_berry_phase(eigenstates, scaling_factor)
    analysis_results['phi_berry_phase'] = berry_phase
    
    return analysis_results

def format_metrics_for_display(result):
    """
    Extract and format metrics from simulation results for display.
    
    Parameters:
    -----------
    result : object
        Simulation result object
    
    Returns:
    --------
    dict
        Dictionary of formatted metrics
    """
    metrics = {}
    
    # Extract basic metrics
    if hasattr(result, 'states') and result.states:
        # Calculate purity
        final_state = result.states[-1]
        if final_state.isket:
            final_dm = final_state * final_state.dag()
        else:
            final_dm = final_state
        
        metrics['purity'] = (final_dm * final_dm).tr().real
        
        # Calculate fidelity to initial state
        initial_state = result.states[0]
        if initial_state.isket:
            initial_dm = initial_state * initial_state.dag()
        else:
            initial_dm = initial_state
            
        metrics['fidelity'] = (initial_dm.dag() * final_dm).tr().real
        
        # Add special metrics for phi-analysis
        if hasattr(result, 'phi_dimension'):
            metrics['phi_dimension'] = result.phi_dimension
            
        if hasattr(result, 'phi_winding'):
            metrics['phi_winding'] = result.phi_winding
            
        if hasattr(result, 'phi_berry_phase'):
            metrics['phi_berry_phase'] = result.phi_berry_phase
    
    return metrics

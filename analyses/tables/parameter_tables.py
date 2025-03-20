"""
Functions for generating parameter overview tables for quantum simulations.
"""

import pandas as pd
import numpy as np
from constants import PHI

def generate_parameter_overview_table(
    system_config=None, 
    simulation_results=None, 
    calculated_metrics=None
) -> pd.DataFrame:
    """
    Generate a comprehensive parameter overview table based on actual system configuration
    and calculated metrics where available, falling back to theoretical ranges when needed.
    
    Parameters:
    -----------
    system_config : dict, optional
        Configuration dictionary with system parameters
    simulation_results : dict, optional
        Dictionary of simulation results
    calculated_metrics : dict, optional
        Dictionary of calculated metrics from simulations
    Returns:
        pd.DataFrame: Table with columns for Symbol, Physical Meaning, 
                     Typical Range/Values, and Units/Dimensions
    """
    parameters = []
    
    # Derive actual qubit range from simulation results if available
    qubit_range = "1-4"  # Default
    if simulation_results:
        try:
            # Extract actual qubit numbers used in simulations
            num_qubits = []
            for result in simulation_results.values():
                if hasattr(result, 'num_qubits'):
                    num_qubits.append(result.num_qubits)
                elif isinstance(result, dict) and 'num_qubits' in result:
                    num_qubits.append(result['num_qubits'])
            
            if num_qubits:
                min_qubits = min(num_qubits)
                max_qubits = max(num_qubits)
                qubit_range = f"{min_qubits}-{max_qubits}"
        except (AttributeError, KeyError, TypeError) as e:
            print(f"Warning: Could not extract qubit range from results: {e}")
    
    # System parameters
    parameters.append({
        "Symbol": "n",
        "Physical Meaning": "Number of qubits",
        "Typical Range/Values": qubit_range,
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "φ",
        "Physical Meaning": "Golden ratio (scaling factor)",
        "Typical Range/Values": f"{PHI:.6f}",
        "Units/Dimensions": "dimensionless"
    })
    
    # Determine actual Hamiltonian types used
    hamiltonian_types = "Σᵢ σᶻᵢ"  # Default
    if system_config and 'hamiltonian_types' in system_config:
        hamiltonian_types = ", ".join(system_config['hamiltonian_types'])
    
    # Hamiltonian parameters
    parameters.append({
        "Symbol": "H₀",
        "Physical Meaning": "Base Hamiltonian",
        "Typical Range/Values": hamiltonian_types,
        "Units/Dimensions": "energy"
    })
    
    unavailable_text = "N/A"
    
    # Determine actual scaling factor range from simulation results
    fs_range = unavailable_text  # Default
    if simulation_results:
        try:
            scaling_factors = []
            # Extract scaling factors from results
            for result in simulation_results.values():
                if hasattr(result, 'scaling_factor'):
                    scaling_factors.append(result.scaling_factor)
                elif isinstance(result, dict) and 'scaling_factor' in result:
                    scaling_factors.append(result['scaling_factor'])
                
            if scaling_factors:
                min_fs = min([fs for fs in scaling_factors if fs is not None])
                max_fs = max([fs for fs in scaling_factors if fs is not None])
                fs_range = f"{min_fs:.1f}-{max_fs:.1f}"
        except (AttributeError, KeyError, TypeError) as e:
            print(f"Warning: Could not extract scaling factor range: {e}")
    
    parameters.append({
        "Symbol": "f_s",
        "Physical Meaning": "Scaling factor",
        "Typical Range/Values": fs_range,
        "Units/Dimensions": "dimensionless"
    })
    
    # Determine actual noise parameters if available
    t1_range = unavailable_text
    t2_range = unavailable_text
    
    if system_config and 'noise_parameters' in system_config:
        noise_params = system_config['noise_parameters']
        if 't1_range' in noise_params:
            t1_range = f"{noise_params['t1_range'][0]}-{noise_params['t1_range'][1]}"
        if 't2_range' in noise_params:
            t2_range = f"{noise_params['t2_range'][0]}-{noise_params['t2_range'][1]}"
    
    # Noise parameters
    parameters.append({
        "Symbol": "T₁",
        "Physical Meaning": "Relaxation time",
        "Typical Range/Values": t1_range,
        "Units/Dimensions": "time units"
    })
    
    parameters.append({
        "Symbol": "T₂",
        "Physical Meaning": "Dephasing time",
        "Typical Range/Values": t2_range,
        "Units/Dimensions": "time units"
    })
    
    # Determine actual fractal dimensions from calculations if available
    fd_range = unavailable_text
    if calculated_metrics and 'fractal_dimensions' in calculated_metrics:
        fd_metrics = calculated_metrics['fractal_dimensions']
        if isinstance(fd_metrics, list) and fd_metrics:
            # Filter out NaN and None values
            valid_fds = [fd for fd in fd_metrics if fd is not None and not np.isnan(fd)]
            if valid_fds:
                min_fd = min(valid_fds)
                max_fd = max(valid_fds)
                fd_range = f"{min_fd:.2f}-{max_fd:.2f}"
    
    # Fractal parameters
    parameters.append({
        "Symbol": "D",
        "Physical Meaning": "Fractal dimension",
        "Typical Range/Values": fd_range,
        "Units/Dimensions": "dimensionless"
    })
    
    # Determine actual winding numbers and berry phases from calculations
    w_values = unavailable_text
    z2_values = unavailable_text
    
    if calculated_metrics and 'topological_invariants' in calculated_metrics:
        topo_metrics = calculated_metrics['topological_invariants']
        if 'winding_numbers' in topo_metrics and topo_metrics['winding_numbers']:
            w_values_found = set([int(w) for w in topo_metrics['winding_numbers'] 
                               if w is not None and not np.isnan(w)])
            if w_values_found:
                w_values = ", ".join([str(w) for w in sorted(w_values_found)])
        
        if 'z2_indices' in topo_metrics and topo_metrics['z2_indices']:
            z2_values_found = set([int(z) for z in topo_metrics['z2_indices'] 
                                if z is not None and not np.isnan(z)])
            if z2_values_found:
                z2_values = ", ".join([str(z) for z in sorted(z2_values_found)])
    
    # Topological parameters
    parameters.append({
        "Symbol": "C",
        "Physical Meaning": "Chern number",
        "Typical Range/Values": w_values,  # Using same range as winding number as approximation
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "ν",
        "Physical Meaning": "Winding number",
        "Typical Range/Values": w_values,
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "Z₂",
        "Physical Meaning": "Z₂ topological index",
        "Typical Range/Values": z2_values,
        "Units/Dimensions": "dimensionless"
    })
    
    # Fibonacci anyon parameters (theoretical/fixed)
    parameters.append({
        "Symbol": "τ",
        "Physical Meaning": "Fibonacci anyon",
        "Typical Range/Values": f"{(1 + np.sqrt(5))/2:.4f}",  # Actual value of golden ratio
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "F",
        "Physical Meaning": "F-matrix for anyon fusion",
        "Typical Range/Values": "2×2 matrix",
        "Units/Dimensions": "dimensionless"
    })
    
    # Determine actual entropy ranges from calculations if available
    s_range = unavailable_text
    if calculated_metrics and 'entanglement_metrics' in calculated_metrics:
        ent_metrics = calculated_metrics['entanglement_metrics']
        if 'entropy_values' in ent_metrics and ent_metrics['entropy_values']:
            valid_s = [s for s in ent_metrics['entropy_values'] 
                      if s is not None and not np.isnan(s)]
            if valid_s:
                min_s = min(valid_s)
                max_s = max(valid_s)
                s_range = f"{min_s:.2f}-{max_s:.2f}"
    
    # Quantum metrics
    parameters.append({
        "Symbol": "S",
        "Physical Meaning": "von Neumann entropy",
        "Typical Range/Values": s_range,
        "Units/Dimensions": "dimensionless"
    })
    
    # Coherence range
    c_range = unavailable_text
    if calculated_metrics and 'coherence_metrics' in calculated_metrics:
        coh_metrics = calculated_metrics['coherence_metrics']
        if 'l1_norm_values' in coh_metrics and coh_metrics['l1_norm_values']:
            valid_c = [c for c in coh_metrics['l1_norm_values'] 
                      if c is not None and not np.isnan(c)]
            if valid_c:
                min_c = min(valid_c)
                max_c = max(valid_c)
                c_range = f"{min_c:.2f}-{max_c:.2f}"
                
    parameters.append({
        "Symbol": "C_l1",
        "Physical Meaning": "l1-norm coherence",
        "Typical Range/Values": c_range,
        "Units/Dimensions": "dimensionless"
    })
    
    return pd.DataFrame(parameters)

def generate_simulation_parameters_table(result) -> pd.DataFrame:
    """
    Generate a table of parameters used in a specific simulation.
    
    Parameters:
        result: Simulation result object
        
    Returns:
        pd.DataFrame: Table with columns for Parameter, Value, and Units
    """
    if result is None:
        return pd.DataFrame()
    
    parameters = []
    
    # Extract parameters from result object
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            # Skip large arrays and objects and private attributes
            if key in ['states', 'times', 'hamiltonian'] or key.startswith('_'):
                continue
                
            # Format value based on type
            if isinstance(value, np.ndarray):
                formatted_value = f"Array of shape {value.shape}"
            elif isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            parameters.append({
                "Parameter": key,
                "Value": formatted_value,
                "Units": "N/A"  # Default, could be improved with parameter-specific units
            })
    
    return pd.DataFrame(parameters)

def export_table_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Export a DataFrame to LaTeX table format.
    
    Parameters:
        df: DataFrame to export
        caption: Table caption
        label: Table label for cross-referencing
        
    Returns:
        str: LaTeX table code
    """
    # Replace special characters in column names
    df = df.copy()
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, caption=caption, label=label)
    
    # Add additional formatting
    latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[htbp]')
    
    return latex_table

"""
Functions for generating parameter overview tables for quantum simulations.
"""

import pandas as pd
import numpy as np
from constants import PHI

def generate_parameter_overview_table() -> pd.DataFrame:
    """
    Generate a comprehensive parameter overview table.
    
    Returns:
        pd.DataFrame: Table with columns for Symbol, Physical Meaning, 
                     Typical Range/Values, and Units/Dimensions
    """
    parameters = []
    
    # System parameters
    parameters.append({
        "Symbol": "n",
        "Physical Meaning": "Number of qubits",
        "Typical Range/Values": "1-4",
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "φ",
        "Physical Meaning": "Golden ratio (scaling factor)",
        "Typical Range/Values": f"{PHI:.6f}",
        "Units/Dimensions": "dimensionless"
    })
    
    # Hamiltonian parameters
    parameters.append({
        "Symbol": "H₀",
        "Physical Meaning": "Base Hamiltonian",
        "Typical Range/Values": "Σᵢ σᶻᵢ",
        "Units/Dimensions": "energy"
    })
    
    parameters.append({
        "Symbol": "f_s",
        "Physical Meaning": "Scaling factor",
        "Typical Range/Values": "0.5-3.0",
        "Units/Dimensions": "dimensionless"
    })
    
    # Noise parameters
    parameters.append({
        "Symbol": "T₁",
        "Physical Meaning": "Relaxation time",
        "Typical Range/Values": "10-100",
        "Units/Dimensions": "time units"
    })
    
    parameters.append({
        "Symbol": "T₂",
        "Physical Meaning": "Dephasing time",
        "Typical Range/Values": "1-10",
        "Units/Dimensions": "time units"
    })
    
    # Fractal parameters
    parameters.append({
        "Symbol": "D",
        "Physical Meaning": "Fractal dimension",
        "Typical Range/Values": "1.0-2.0",
        "Units/Dimensions": "dimensionless"
    })
    
    # Topological parameters
    parameters.append({
        "Symbol": "C",
        "Physical Meaning": "Chern number",
        "Typical Range/Values": "0, ±1, ±2",
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "ν",
        "Physical Meaning": "Winding number",
        "Typical Range/Values": "0, ±1, ±2",
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "Z₂",
        "Physical Meaning": "Z₂ topological index",
        "Typical Range/Values": "0, 1",
        "Units/Dimensions": "dimensionless"
    })
    
    # Fibonacci anyon parameters
    parameters.append({
        "Symbol": "τ",
        "Physical Meaning": "Fibonacci anyon",
        "Typical Range/Values": "N/A",
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "F",
        "Physical Meaning": "F-matrix for anyon fusion",
        "Typical Range/Values": "2×2 matrix",
        "Units/Dimensions": "dimensionless"
    })
    
    # Quantum metrics
    parameters.append({
        "Symbol": "S",
        "Physical Meaning": "von Neumann entropy",
        "Typical Range/Values": "0-log(d)",
        "Units/Dimensions": "dimensionless"
    })
    
    parameters.append({
        "Symbol": "C_l1",
        "Physical Meaning": "l1-norm coherence",
        "Typical Range/Values": "0-1",
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

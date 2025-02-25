"""
Functions for calculating quantum coherence measures.
"""

import numpy as np
from qutip import Qobj
from typing import Union, List

def l1_coherence(state: Qobj, dim: int = None) -> float:
    """
    Calculate the l1-norm of coherence for a quantum state.
    For a density matrix ρ, this is the sum of absolute values of off-diagonal elements.
    
    Parameters:
        state: Quantum state (Qobj)
        dim: Dimension of the system (optional)
        
    Returns:
        float: L1 coherence measure
    """
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get matrix representation
    matrix = rho.full()
    n = matrix.shape[0]
    
    # Sum absolute values of off-diagonal elements
    coherence = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                coherence += abs(matrix[i,j])
    
    # Normalize by maximum possible coherence (n(n-1) for n-dimensional system)
    max_coherence = n * (n-1)
    return coherence / max_coherence if max_coherence > 0 else 0.0

def relative_entropy_coherence(state: Qobj) -> float:
    """
    Calculate the relative entropy of coherence.
    This is the difference between von Neumann entropy of the dephased state
    and the original state.
    
    Parameters:
        state: Quantum state (Qobj)
        
    Returns:
        float: Relative entropy coherence measure
    """
    from .entropy import von_neumann_entropy
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get diagonal state (dephased)
    diag_elements = np.diag(rho.full())
    diag_state = Qobj(np.diag(diag_elements))
    
    # Calculate relative entropy
    return von_neumann_entropy(diag_state) - von_neumann_entropy(rho)

def robustness_coherence(state: Qobj) -> float:
    """
    Calculate the robustness of coherence.
    This quantifies how much mixing with another state is needed to destroy coherence.
    
    Parameters:
        state: Quantum state (Qobj)
        
    Returns:
        float: Robustness of coherence measure in [0,1]
    """
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get matrix representation
    matrix = rho.full()
    
    # Calculate sum of absolute values of off-diagonal elements
    off_diag_sum = np.sum(np.abs(matrix)) - np.sum(np.abs(np.diag(matrix)))
    
    # Normalize by dimension
    dim = matrix.shape[0]
    return off_diag_sum / (2 * (dim - 1)) if dim > 1 else 0.0

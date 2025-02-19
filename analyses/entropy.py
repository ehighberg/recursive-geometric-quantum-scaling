"""
Functions for calculating quantum entropy measures.
"""

import numpy as np
from qutip import Qobj
from typing import Union, List

def von_neumann_entropy(state: Union[Qobj, List[Qobj]]) -> float:
    """
    Calculate the von Neumann entropy of a quantum state.
    S = -Tr(ρ log_2(ρ))
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        
    Returns:
        float: von Neumann entropy value
    """
    if isinstance(state, list):
        return [von_neumann_entropy(s) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get eigenvalues
    eigs = np.real(np.linalg.eigvals(rho.full()))
    
    # Remove small negative eigenvalues (numerical artifacts)
    eigs = eigs[eigs > 1e-15]
    
    # Calculate entropy
    entropy = 0.0
    for p in eigs:
        if p > 0:  # Avoid log(0)
            entropy -= p * np.log2(p)
    
    # Normalize by maximum entropy (log2(d) for d-dimensional system)
    max_entropy = np.log2(len(eigs))
    return float(entropy / max_entropy if max_entropy > 0 else 0.0)

def renyi_entropy(state: Union[Qobj, List[Qobj]], alpha: float = 2.0) -> float:
    """
    Calculate the Rényi entropy of order alpha.
    S_α = 1/(1-α) log_2(Tr(ρ^α))
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        alpha: Order of Rényi entropy (α > 0, α ≠ 1)
        
    Returns:
        float: Rényi entropy value
    """
    if isinstance(state, list):
        return [renyi_entropy(s, alpha) for s in state]
    
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if abs(alpha - 1.0) < 1e-10:
        return von_neumann_entropy(state)
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get eigenvalues
    eigs = np.real(np.linalg.eigvals(rho.full()))
    eigs = eigs[eigs > 1e-15]  # Remove numerical noise
    
    # Calculate Rényi entropy
    return float(1.0 / (1.0 - alpha) * np.log2(np.sum(eigs**alpha)))

def linear_entropy(state: Union[Qobj, List[Qobj]]) -> float:
    """
    Calculate the linear entropy (special case of Rényi entropy with α=2).
    S_L = 1 - Tr(ρ^2)
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        
    Returns:
        float: Linear entropy value in [0,1]
    """
    if isinstance(state, list):
        return [linear_entropy(s) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Calculate purity Tr(ρ^2)
    purity = (rho * rho).tr().real
    
    # Calculate linear entropy
    d = rho.shape[0]  # dimension of Hilbert space
    return float((1.0 - purity) * d/(d-1)) if d > 1 else 0.0

def tsallis_entropy(state: Union[Qobj, List[Qobj]], q: float = 2.0) -> float:
    """
    Calculate the Tsallis entropy.
    S_q = (1-Tr(ρ^q))/(q-1)
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        q: Entropic index (q > 0, q ≠ 1)
        
    Returns:
        float: Tsallis entropy value
    """
    if isinstance(state, list):
        return [tsallis_entropy(s, q) for s in state]
    
    if q <= 0:
        raise ValueError("q must be positive")
    if abs(q - 1.0) < 1e-10:
        return von_neumann_entropy(state)
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get eigenvalues
    eigs = np.real(np.linalg.eigvals(rho.full()))
    eigs = eigs[eigs > 1e-15]  # Remove numerical noise
    
    # Calculate Tsallis entropy
    return float((1.0 - np.sum(eigs**q)) / (q - 1.0))

"""
Functions for calculating quantum entanglement measures.
"""

import numpy as np
from qutip import Qobj, partial_transpose, entropy_vn, tensor
from typing import Union, List

def concurrence(state: Union[Qobj, List[Qobj]]) -> float:
    """
    Calculate the concurrence for a two-qubit state.
    For pure states: C = sqrt(2(1-Tr(ρ_A^2)))
    For mixed states: C = max(0, λ1 - λ2 - λ3 - λ4)
    where λi are eigenvalues of sqrt(ρ(σy⊗σy)ρ*(σy⊗σy)) in decreasing order.
    
    Parameters:
        state: Two-qubit quantum state (Qobj) or list of states
        
    Returns:
        float: Concurrence value in [0,1]
    """
    if isinstance(state, list):
        return [concurrence(s) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Check dimensions for two-qubit states
    if not (len(rho.dims[0]) == 2 and all(d == 2 for d in rho.dims[0])):
        raise ValueError("Concurrence is only defined for two-qubit states")
    
    # For pure states, use simpler formula
    if rho.tr() == 1 and (rho * rho).tr().real - 1 < 1e-10:
        # Get reduced density matrix
        rho_A = rho.ptrace(0)
        return float(np.sqrt(2 * (1 - (rho_A * rho_A).tr().real)))
    
    # For mixed states, use eigenvalue formula
    Y = Qobj([[0, -1j], [1j, 0]])  # Pauli Y
    sigma_y = tensor([Y, Y])
    R = rho * sigma_y * rho.conj() * sigma_y
    
    # Get eigenvalues in decreasing order
    evals = np.sqrt(np.real(np.linalg.eigvals(R.full())))
    evals = np.sort(evals)[::-1]
    
    # Calculate concurrence
    c = float(max(0, evals[0] - evals[1] - evals[2] - evals[3]))
    return c

def negativity(state: Union[Qobj, List[Qobj]]) -> float:
    """
    Calculate the negativity of a bipartite quantum state.
    N = (||ρ^(TA)||_1 - 1)/2 where ρ^(TA) is partial transpose.
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        
    Returns:
        float: Negativity value in [0,1]
    """
    if isinstance(state, list):
        return [negativity(s) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Calculate partial transpose
    # For single-qubit states, return 0 since there's no entanglement
    if len(rho.dims[0]) == 1:
        return 0.0
        
    # Handle different dimensional cases systematically
    system_dims = rho.dims[0]
    num_qubits = len(system_dims)
    
    if num_qubits == 1:
        return 0.0  # No entanglement for single qubit
    
    # For multi-qubit systems, we'll use the first qubit as subsystem A
    # and the rest as subsystem B
    if not all(d == 2 for d in system_dims):
        raise ValueError("Negativity calculation currently only supports qubit systems")
    
    # Create mask for partial transpose
    # True for the first qubit (subsystem A), False for the rest
    mask = [True] + [False] * (num_qubits - 1)
    
    # Perform partial transpose
    try:
        rho_pt = partial_transpose(rho, mask)
    except ValueError:
        # If partial transpose fails, try converting to full matrix first
        rho_full = rho.full()
        rho_qobj = Qobj(rho_full, dims=rho.dims)
        rho_pt = partial_transpose(rho_qobj, mask)
    
    # Calculate trace norm
    eigs = np.linalg.eigvals(rho_pt.full())
    trace_norm = np.sum(np.abs(eigs))
    
    # Calculate negativity
    return float((trace_norm - 1) / 2)

def log_negativity(state: Union[Qobj, List[Qobj]]) -> float:
    """
    Calculate the logarithmic negativity of a bipartite quantum state.
    EN = log2(||ρ^(TA)||_1)
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        
    Returns:
        float: Logarithmic negativity value
    """
    if isinstance(state, list):
        return [log_negativity(s) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # For single-qubit states, return 0 since there's no entanglement
    if len(rho.dims[0]) == 1:
        return 0.0
        
    # Handle different dimensional cases systematically
    system_dims = rho.dims[0]
    num_qubits = len(system_dims)
    
    if num_qubits == 1:
        return 0.0  # No entanglement for single qubit
    
    # For multi-qubit systems, we'll use the first qubit as subsystem A
    # and the rest as subsystem B
    if not all(d == 2 for d in system_dims):
        raise ValueError("Logarithmic negativity calculation currently only supports qubit systems")
    
    # Create mask for partial transpose
    # True for the first qubit (subsystem A), False for the rest
    mask = [True] + [False] * (num_qubits - 1)
    
    # Perform partial transpose
    try:
        rho_pt = partial_transpose(rho, mask)
    except ValueError:
        # If partial transpose fails, try converting to full matrix first
        rho_full = rho.full()
        rho_qobj = Qobj(rho_full, dims=rho.dims)
        rho_pt = partial_transpose(rho_qobj, mask)
    
    # Calculate trace norm
    eigs = np.linalg.eigvals(rho_pt.full())
    trace_norm = np.sum(np.abs(eigs))
    
    # Calculate logarithmic negativity
    return float(np.log2(trace_norm))

def entanglement_entropy(state: Union[Qobj, List[Qobj]], subsys: int = 0) -> float:
    """
    Calculate the entanglement entropy of a bipartite quantum state.
    S = -Tr(ρ_A log_2(ρ_A)) where ρ_A is reduced density matrix.
    
    Parameters:
        state: Quantum state (Qobj) or list of states
        subsys: Which subsystem to trace out (0 or 1)
        
    Returns:
        float: Entanglement entropy value
    """
    if isinstance(state, list):
        return [entanglement_entropy(s, subsys) for s in state]
    
    # Convert to density matrix if needed
    if state.isket:
        rho = state * state.dag()
    else:
        rho = state
    
    # Get reduced density matrix
    rho_A = rho.ptrace(subsys)
    
    # Calculate von Neumann entropy
    return float(entropy_vn(rho_A))

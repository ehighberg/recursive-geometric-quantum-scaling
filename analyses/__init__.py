"""
Module for quantum analyses including entanglement, entropy, and coherence metrics.
"""

from .entanglement import compute_negativity, compute_log_negativity, concurrence
from .entropy import compute_vn_entropy, compute_linear_entropy, compute_mutual_information
from .coherence import l1_coherence
from qutip import fidelity
from . import scaling

def run_analyses(initial_state, current_state):
    """
    Computes entanglement, entropy, coherence, and fidelity measures between the initial and current quantum states.
    
    Parameters:
        initial_state (Qobj): The initial quantum state.
        current_state (Qobj): The current quantum state after evolution.
    
    Returns:
        dict: Dictionary containing various quantum metrics including fidelity.
    """
    # Convert states to density matrices if they are kets
    if initial_state.isket:
        rho_init = initial_state * initial_state.dag()
    else:
        rho_init = initial_state
    
    if current_state.isket:
        rho_current = current_state * current_state.dag()
    else:
        rho_current = current_state
    
    fid = fidelity(rho_init, rho_current)
    
    neg_val = compute_negativity(rho_current, sysA=[0])
    vn_ent = compute_vn_entropy(rho_current, base=2)
    co_val = l1_coherence(rho_current, dim=rho_current.shape[0])
    purity = (rho_current * rho_current).tr().real
    
    results = {
        "negativity": neg_val,
        "vn_entropy": vn_ent,
        "l1_coherence": co_val,
        "purity": purity,
        "fidelity": fid
    }
    
    # Calculate appropriate entanglement measures based on number of qubits
    num_qubits = len(current_state.dims[0])
    if num_qubits == 2:
        # For two-qubit states, use concurrence
        results.update({
            'concurrence': concurrence(current_state)
        })
    elif num_qubits > 2:
        # For multi-qubit states, use negativity and log_negativity
        results.update({
            'negativity': compute_negativity(current_state, sysA=[0]),
            'log_negativity': compute_log_negativity(current_state)
        })
    
    return results

__all__ = [
    'l1_coherence',
    'concurrence',
    'compute_negativity',
    'compute_log_negativity',
    'compute_vn_entropy',
    'compute_linear_entropy',
    'compute_mutual_information',
    'run_analyses',
    'scaling'
]

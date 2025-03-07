"""
Module for quantum analyses including entanglement, entropy, and coherence metrics.
"""

from .entanglement import negativity, log_negativity, concurrence
from .entropy import von_neumann_entropy, linear_entropy, compute_mutual_information
from .coherence import l1_coherence
from .fractal_analysis import compute_energy_spectrum, estimate_fractal_dimension
from .topological_invariants import compute_chern_number, compute_winding_number, compute_z2_index
from qutip import fidelity
from . import scaling
from . import tables

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
    
    neg_val = negativity(rho_current, sysA=[0])
    vn_ent = von_neumann_entropy(rho_current, base=2)
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
            'negativity': negativity(current_state, sysA=[0]),
            'log_negativity': log_negativity(current_state)
        })
    
    return results

__all__ = [
    'l1_coherence',
    'concurrence',
    'negativity',
    'log_negativity',
    'von_neumann_entropy',
    'linear_entropy',
    'compute_mutual_information',
    'compute_energy_spectrum',
    'estimate_fractal_dimension',
    'compute_chern_number',
    'compute_winding_number',
    'compute_z2_index',
    'run_analyses',
    'scaling',
    'tables'
]

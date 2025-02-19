from .entanglement import negativity, log_negativity, concurrence
from .entropy import von_neumann_entropy, renyi_entropy
from .coherence import coherence_metric
from qutip import fidelity

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
    
    neg_val = negativity(rho_current)
    vn_ent = von_neumann_entropy(rho_current)
    co_val = coherence_metric(rho_current)
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
            'negativity': negativity(current_state),
            'log_negativity': log_negativity(current_state)
        })
    
    return results

__all__ = [
    'coherence_metric',
    'concurrence',
    'negativity',
    'log_negativity',
    'von_neumann_entropy',
    'renyi_entropy',
    'run_analyses'
]

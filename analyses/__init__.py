from .entanglement import compute_negativity, compute_log_negativity, bipartite_partial_trace
from .entropy import compute_vn_entropy, compute_linear_entropy, compute_mutual_information
from .coherence import l1_coherence, relative_entropy_coherence
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
    
    neg_val = compute_negativity(rho_current)
    vn_ent = compute_vn_entropy(rho_current)
    co_val = l1_coherence(rho_current)
    purity = (rho_current * rho_current).tr().real
    
    return {
        "negativity": neg_val,
        "vn_entropy": vn_ent,
        "l1_coherence": co_val,
        "purity": purity,
        "fidelity": fid
    }

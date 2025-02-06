from .entanglement import compute_negativity, compute_log_negativity, bipartite_partial_trace
from .entropy import compute_vn_entropy, compute_linear_entropy, compute_mutual_information
from .coherence import l1_coherence, relative_entropy_coherence

def run_analyses(rho):
    """
    Computes entanglement, entropy, and coherence measures for a given quantum state rho.
    """
    neg_val = compute_negativity(rho)
    vn_ent = compute_vn_entropy(rho)
    co_val = l1_coherence(rho)
    return {
        "negativity": neg_val,
        "vn_entropy": vn_ent,
        "l1_coherence": co_val,
    }

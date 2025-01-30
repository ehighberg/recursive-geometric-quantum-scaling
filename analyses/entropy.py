# analyses/entropy.py

"""
Entropy-related analysis using QuTiP:
 - von Neumann entropy
 - linear entropy
 - mutual information, etc.
"""

from qutip import entropy_vn, entropy_linear, ptrace, Qobj

def compute_vn_entropy(rho, base=2):
    """
    von Neumann entropy S(ρ) = -Tr(ρ log ρ).
    QuTiP: entropy_vn(rho, base=2).
    """
    return entropy_vn(rho, base=base)

def compute_linear_entropy(rho):
    """
    Linear entropy: S_L(ρ) = 1 - Tr(ρ^2).
    QuTiP: entropy_linear(rho).
    """
    return entropy_linear(rho)

def compute_mutual_information(rho, subsysA, subsysB, dims=None):
    """
    Example: bipartite mutual information I(A:B) = S(A) + S(B) - S(A,B)
    Using partial trace for each subsystem.
    'dims' define the total dimension structure if not 2 qubits.
    """
    # For a 2-qubit system, default dims=(2,2).
    if dims is None:
        dims = [2,2]

    # partial trace
    rhoA = ptrace(rho, subsysA)
    rhoB = ptrace(rho, subsysB)

    # von Neumann entropies
    S_AB = entropy_vn(rho)
    S_A  = entropy_vn(rhoA)
    S_B  = entropy_vn(rhoB)
    return (S_A + S_B - S_AB)

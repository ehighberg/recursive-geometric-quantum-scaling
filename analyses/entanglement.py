# analyses/entanglement.py

"""
Entanglement analysis functions using QuTiP.
Examples:
 - negativity
 - logarithmic negativity
 - partial trace usage to get bipartite entanglement measures
"""

 
import numpy as np
from qutip import partial_transpose, ptrace, negativity
from qutip import Qobj

def compute_negativity(rho, sysA_dims=None):
    """
    Computes the negativity for a bipartite system described by rho.
    If sysA_dims is None, assume a 2-qubit partition (2,2).
    For multi-qubit, pass 'sysA_dims' to define the dimension of subsystem A.
    
    QuTiP's negativity function:
      negativity(rho, sys A dimension)
    if not isinstance(rho, Qobj) or not rho.isoper or not rho.isherm:
        raise ValueError("Input must be a density matrix (Qobj operator).")

    """
    if sysA_dims is None:
        # default: 2-qubit => first qubit is sysA => dims=(2,2)
        # Calculate negativity for bipartite system
        neg_val = negativity(rho, [2,2])
    else:
        # For multi-qubit systems
        neg_val = negativity(rho, sysA_dims)
    return neg_val

def compute_log_negativity(rho, sysA_dims=None):
    """
    Computes the log-negativity for a bipartite system.
    If sysA_dims is None, assume 2-qubit partition (2,2).
    """
    if sysA_dims is None:
        neg = compute_negativity(rho)
    else:
        neg = compute_negativity(rho, sysA_dims)
    # Logarithmic negativity formula: log2(2*negativity + 1)
    ln_val = np.log2(2 * neg + 1)
    return ln_val

def bipartite_partial_trace(rho, keep=0, dims=[2,2]):
    """
    Example: partial trace to keep subsystem 'keep' in a 2-qubit system => reduced density matrix
    """
    # QuTiP's ptrace usage:
    return ptrace(rho, keep)

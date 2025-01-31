# analyses/entanglement.py

"""
Entanglement analysis functions using QuTiP.
Examples:
 - negativity
 - logarithmic negativity
 - partial trace usage to get bipartite entanglement measures
"""

from qutip import partial_transpose, ptrace, negativity, log_negativity
from qutip import Qobj

def compute_negativity(rho, sysA_dims=None):
    """
    Computes the negativity for a bipartite system described by rho.
    If sysA_dims is None, assume a 2-qubit partition (2,2).
    For multi-qubit, pass 'sysA_dims' to define the dimension of subsystem A.
    
    QuTiP's negativity function:
      negativity(rho, sys A dimension)
    """
    if sysA_dims is None:
        # default: 2-qubit => first qubit is sysA => dims=(2,2)
        # negativity(rho, [2,2], sys_A=0)
        neg_val = negativity(rho, [2,2])
    else:
        # e.g. negativity(rho, sysA_dims, sys_A=0)
        neg_val = negativity(rho, sysA_dims)
    return neg_val

def compute_log_negativity(rho, sysA_dims=None):
    """
    Computes the log-negativity for a bipartite system.
    If sysA_dims is None, assume 2-qubit partition (2,2).
    """
    if sysA_dims is None:
        ln_val = log_negativity(rho, [2,2])
    else:
        ln_val = log_negativity(rho, sysA_dims)
    return ln_val

def bipartite_partial_trace(rho, keep=0, dims=[2,2]):
    """
    Example: partial trace to keep subsystem 'keep' in a 2-qubit system => reduced density matrix
    """
    # QuTiP's ptrace usage:
    return ptrace(rho, keep)

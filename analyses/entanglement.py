# analyses/entanglement.py

"""
Entanglement analysis functions using QuTiP.
Examples:
 - negativity
 - logarithmic negativity
 - partial trace usage to get bipartite entanglement measures
"""

import numpy as np
from qutip import partial_transpose, ptrace, Qobj


def compute_negativity(rho, sysA=None):
    """
    Computes the negativity for a bipartite system described by rho.
    If sysA is None, assume the first subsystem.
    For multi-qubit, pass 'sysA' as a list of subsystem indices to perform partial transpose over.

    Parameters:
        rho (Qobj): Density matrix.
        sysA (list or int, optional): Subsystem indices to perform partial transpose over.
                                     Defaults to transposing over the first subsystem.

    Returns:
        float: The negativity of the bipartite system.
    """
    if not isinstance(rho, Qobj) or not rho.isoper or not rho.isherm:
        raise ValueError("Input must be a density matrix (Qobj operator).")
    
    # Handle sysA=None by assuming first subsystem
    if sysA is None:
        sysA = [0]
    elif isinstance(sysA, int):
        sysA = [sysA]
    elif isinstance(sysA, list):
        sysA = sysA
    else:
        raise ValueError("sysA must be None, an integer, or a list of integers.")
    
    # Validate sysA indices
    num_subsystems = len(rho.dims[0])
    if any(idx >= num_subsystems or idx < 0 for idx in sysA):
        raise ValueError("sysA contains invalid subsystem indices.")
    
    # Create mask array matching the number of subsystems
    mask = np.zeros(num_subsystems, dtype=int)
    mask[sysA] = 1  # Set 1 for subsystems to transpose
    
    # Perform partial transpose with exception handling
    try:
        rho_pt = partial_transpose(rho, mask)
    except Exception as e:
        raise ValueError(f"partial_transpose failed: {e}")
    
    # Compute eigenvalues
    eigenvals = rho_pt.eigenenergies()
    
    # The negativity is the sum of absolute values of negative eigenvalues
    neg_eigenvals = [e for e in eigenvals if e < 0]
    if not neg_eigenvals:
        return 0.0
    
    neg_val = sum(abs(e) for e in neg_eigenvals)
    
    return neg_val


def compute_log_negativity(rho, sysA=None):
    """
    Computes the log-negativity for a bipartite system.
    If sysA is None, assume the first subsystem.
    
    Logarithmic negativity formula: ln(2 * negativity + 1)
    
    Parameters:
        rho (Qobj): Density matrix.
        sysA (list or int, optional): Subsystem indices to perform partial transpose over.
                                     Defaults to transposing over the first subsystem.
    
    Returns:
        float: The log-negativity of the bipartite system.
    """
    if not isinstance(rho, Qobj) or not rho.isoper or not rho.isherm:
        raise ValueError("Input must be a density matrix (Qobj operator).")
    
    # Handle sysA=None by assuming first subsystem
    if sysA is None:
        sysA = [0]
    elif isinstance(sysA, int):
        sysA = [sysA]
    elif isinstance(sysA, list):
        sysA = sysA
    else:
        raise ValueError("sysA must be None, an integer, or a list of integers.")
    
    # Validate sysA indices
    num_subsystems = len(rho.dims[0])
    if any(idx >= num_subsystems or idx < 0 for idx in sysA):
        raise ValueError("sysA contains invalid subsystem indices.")
    
    # Compute negativity
    try:
        neg = compute_negativity(rho, sysA)
    except ValueError as e:
        raise ValueError(f"compute_negativity failed: {e}")
    
    # Logarithmic negativity formula: ln(2 * negativity + 1)
    ln_val = np.log(2 * neg + 1)
    return ln_val


def bipartite_partial_trace(rho, keep=0, dims=[2,2]):
    """
    Example: Partial trace to keep subsystem 'keep' in a 2-qubit system => reduced density matrix.
    
    Parameters:
        rho (Qobj): Density matrix of the composite system.
        keep (int): Index of the subsystem to keep.
        dims (list): Dimensions of each subsystem.
    
    Returns:
        Qobj: Reduced density matrix after partial trace.
    """
    # QuTiP's ptrace usage:
    return ptrace(rho, keep)

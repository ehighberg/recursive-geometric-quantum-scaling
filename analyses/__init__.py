"""
Analysis module for quantum metrics and visualization.
"""
from typing import Dict, Any
from qutip import Qobj

from .coherence import coherence_metric
from .entanglement import concurrence, negativity, log_negativity
from .entropy import von_neumann_entropy, renyi_entropy

def run_analyses(state: Qobj) -> Dict[str, Any]:
    """
    Run all available quantum analyses on a given state.
    
    Parameters:
        state: Quantum state to analyze
        
    Returns:
        Dictionary mapping metric names to their values
    """
    results = {
        'vn_entropy': von_neumann_entropy(state),
        'l1_coherence': coherence_metric(state)
    }
    
    # Calculate appropriate entanglement measures based on number of qubits
    num_qubits = len(state.dims[0])
    if num_qubits == 2:
        # For two-qubit states, use concurrence
        results.update({
            'concurrence': concurrence(state)
        })
    elif num_qubits > 2:
        # For multi-qubit states, use negativity and log_negativity
        results.update({
            'negativity': negativity(state),
            'log_negativity': log_negativity(state)
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

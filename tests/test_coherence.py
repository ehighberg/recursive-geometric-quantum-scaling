# tests/test_coherence.py

import pytest
import numpy as np
from qutip import Qobj, basis, ket2dm
from analyses.coherence import (
    coherence_metric, relative_entropy_coherence
)

def test_l1_coherence_single_qubit_offdiag():
    """
    Single qubit with off-diagonal elements => check L1 measure.
    Example: (|0> + |1>)/sqrt(2) => density matrix => off diag = 0.5
    => L1 = sum_{i!=j} |rho_{i,j}| = 2 * 0.5 = 1.0
    """
    from qutip import Qobj
    # |+> = 1/sqrt(2) (|0> + |1>)
    psi_plus = (basis(2,0) + basis(2,1)).unit()
    rho = ket2dm(psi_plus)
    c_l1 = coherence_metric(rho)
    assert abs(c_l1 - 1.0) < 1e-6, f"L1 coherence ~ 1.0 for single qubit +"

def test_relative_entropy_coherence_single_qubit():
    """
    For single qubit |+>, relative entropy of coherence:
      diag(rho) => diag(0.5, 0.5), S(diag(rho))= ln(2)=0.693..., S(rho)=0 => C_rel= ln(2)=0.693
      (we use natural log in QuTiP).
    """
    from qutip import ket2dm
    plus = (basis(2,0) + basis(2,1)).unit()
    rho = ket2dm(plus)
    rec = relative_entropy_coherence(rho)
    # ln(2) ~ 0.693147
    assert abs(rec - 0.693147) < 1e-4, f"REL. ENT. coherence ~ ln(2). got {rec}"

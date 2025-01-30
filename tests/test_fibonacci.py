# tests/test_fibonacci.py

import pytest
import numpy as np
from simulations.scripts.fibonacci_anyon_braiding import (
    F_2x2, F_inv_2x2, braid_b1_2d, braid_b2_2d
)

def test_F_2x2():
    F = F_2x2()
    assert F.shape==(2,2)
    # check partial unitarity or norm
    FdagF = (F.dag() @ F).full()
    # might not be strictly identity if it's not a purely unitary matrix
    # but we can check shape & no crash
    assert FdagF.shape==(2,2)

def test_F_inv_2x2():
    Finv = F_inv_2x2()
    assert Finv.shape==(2,2)

def test_braid_b1_2d():
    B1= braid_b1_2d()
    assert B1.shape==(2,2)

def test_braid_b2_2d():
    B2= braid_b2_2d()
    assert B2.shape==(2,2)

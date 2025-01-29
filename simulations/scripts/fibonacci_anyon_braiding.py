#!/usr/bin/env python
# simulations/scripts/fibonacci_anyon_braiding.py
"""
Fibonacci anyon braiding demos: 3 anyons (2D) and 4 anyons (3D).
Includes toy error model + measure-and-correct.
"""

import numpy as np
import math
from qutip import Qobj

phi = (1 + np.sqrt(5)) / 2

# 2D subspace for 3 anyons
F_2x2_np = (1/np.sqrt(phi))*np.array([
    [1, np.sqrt(phi)],
    [np.sqrt(phi), -1]
], dtype=complex)
F_inv_2x2_np = np.linalg.inv(F_2x2_np)

R_1   = np.exp(-4.0j*math.pi/5.0)
R_tau = np.exp( 3.0j*math.pi/5.0)

from qutip import Qobj

F_2x2 = Qobj(F_2x2_np, dims=[[2],[2]])
F_inv_2x2 = Qobj(F_inv_2x2_np, dims=[[2],[2]])

def braid_b1_2d():
    mat = np.array([
        [R_1,    0],
        [0,    R_tau]
    ], dtype=complex)
    return Qobj(mat, dims=[[2],[2]])

def braid_b2_2d():
    R_diag = np.array([[R_1,0],[0,R_tau]], dtype=complex)
    R_op = Qobj(R_diag, dims=[[2],[2]])
    return F_inv_2x2*R_op*F_2x2

# Mock 3D for 4 anyons
def mock_F_3x3():
    M = np.eye(3, dtype=complex)
    M[0,1] = 0.3
    M[1,0] = 0.3
    return Qobj(M)

F_3x3 = mock_F_3x3()
F_inv_3x3 = F_3x3.inv()

def R_3x3():
    return Qobj(np.diag([R_1,R_tau,R_tau]), dims=[[3],[3]])

def braid_b1_3d():
    return R_3x3()

def braid_b2_3d():
    Rop = R_3x3()
    return F_inv_3x3 * Rop * F_3x3

def braid_b3_3d():
    Rop = R_3x3()
    return F_3x3 * Rop * F_inv_3x3

def dynamic_braid_3d(psi, measure_outcome):
    B1 = braid_b1_3d()
    B2 = braid_b2_3d()
    B3 = braid_b3_3d()

    if measure_outcome == 0:
        seq = [B2, B3]
    else:
        seq = [B1, B1, B2]

    op = Qobj(np.eye(3, dtype=complex))
    for b in seq:
        op = b*op
    return op*psi

def approximate_phi_op(theta_degrees):
    B1_2 = braid_b1_2d()
    B2_2 = braid_b2_2d()
    approx_lib = {
       36.0: [B2_2, B1_2],
       72.0: [B1_2, B1_2, B1_2]
    }
    angles = list(approx_lib.keys())
    best_angle = min(angles, key=lambda x: abs(x - theta_degrees))
    braids = approx_lib[best_angle]
    total_op = Qobj(np.eye(2, dtype=complex))
    for b in braids:
        total_op = b*total_op
    return total_op

def apply_braid_with_error(psi, braid_op, error_prob):
    new_psi = braid_op*psi
    if np.random.rand()<error_prob:
        dim = psi.shape[0]
        E_mat = np.eye(dim, dtype=complex)
        if dim==2:
            E_mat[0,0]=0
            E_mat[0,1]=1
            E_mat[1,0]=1
        elif dim==3:
            E_mat[0,0]=0
            E_mat[0,1]=1
            E_mat[1,0]=1
        E_op = Qobj(E_mat)
        new_psi = E_op*new_psi
    return new_psi

def measure_and_correct(psi):
    amp0 = abs(psi[0][0])**2
    if amp0<0.5:
        dim = psi.shape[0]
        fix_mat = np.eye(dim, dtype=complex)
        if dim>=2:
            fix_mat[0,0] = 0
            fix_mat[0,1] = 1
            fix_mat[1,0] = 1
        fixer = Qobj(fix_mat)
        psi = fixer*psi
    return psi

if __name__=="__main__":
    # Demo
    ket0_2d = Qobj(np.array([1,0], dtype=complex))
    psi_init_2d = (ket0_2d + Qobj(np.array([0,1],dtype=complex))).unit()

    psi_after_2d = apply_braid_with_error(psi_init_2d, (braid_b1_2d()*braid_b2_2d()), 0.2)
    psi_corr_2d  = measure_and_correct(psi_after_2d)
    print("3 anyons => final state:", psi_corr_2d)

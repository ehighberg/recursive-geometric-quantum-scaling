from .abstract_braid_generator import AbstractBraidGenerator
import numpy as np
from qutip import Qobj

class BraidGenerator2D(AbstractBraidGenerator):
    """
    Braid generator for Fibonacci anyons in a two-dimensional Hilbert space.
    
    Implements the Fibonacci anyon braiding rules with:
      - The F-matrix defined as:
            F = [[φ⁻¹,     φ^(-1/2)],
                 [φ^(-1/2), -φ⁻¹]]
      - The elementary braiding operator for index 1 as a diagonal matrix:
            B₁ = diag(exp(-4πi/5), exp(3πi/5))
      - The braid for index 2 computed via:
            B₂ = F⁻¹ B₁ F
    """
    def F_matrix(self) -> Qobj:
        # Compute φ⁻¹ and φ^(-1/2)
        phi_inv = 1.0 / self.phi
        phi_inv_sqrt = self.phi ** (-0.5)
        mat = np.array([
            [phi_inv,      phi_inv_sqrt],
            [phi_inv_sqrt, -phi_inv]
        ], dtype=complex)
        return Qobj(mat)

    def elementary_braid(self, braid_index: int) -> Qobj:
        if braid_index == 1:
            # Braiding phases for Fibonacci anyons:
            R_1   = np.exp(-4j * np.pi / 5)  # vacuum fusion channel
            R_tau = np.exp(3j * np.pi / 5)     # τ fusion channel
            return Qobj(np.diag([R_1, R_tau]))
        elif braid_index == 2:
            F = self.F_matrix()
            # Compute B₂ = F⁻¹ B₁ F
            return F.inv() @ self.elementary_braid(1) @ F
        else:
            raise ValueError("Unsupported braid index for 2D. Valid indices are 1 or 2.")

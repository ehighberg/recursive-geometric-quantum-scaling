from .abstract_braid_generator import AbstractBraidGenerator
import numpy as np
from qutip import Qobj

class BraidGenerator2D(AbstractBraidGenerator):
    def F_matrix(self) -> Qobj:
        mat = (1.0 / np.sqrt(self.phi)) * np.array([
            [1, np.sqrt(self.phi)],
            [np.sqrt(self.phi), -1]
        ], dtype=complex)
        return Qobj(mat)

    def elementary_braid(self, braid_index: int) -> Qobj:
        if braid_index == 1:
            R_1 = np.exp(-4j * np.pi / 5)
            R_tau = np.exp(3j * np.pi / 5)
            return Qobj(np.diag([R_1, R_tau]))
        elif braid_index == 2:
            # Changed '*' to '@' for matrix multiplication to resolve Pylance error
            return self.F_matrix().inv() @ self.elementary_braid(1) @ self.F_matrix()
        else:
            raise ValueError("Unsupported braid index for 2D.")
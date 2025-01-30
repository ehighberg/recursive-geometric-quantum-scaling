from abc import ABC, abstractmethod
from qutip import Qobj
import numpy as np

class AbstractBraidGenerator(ABC):
    def __init__(self, dimensionality: int):
        self.dim = dimensionality
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    @abstractmethod
    def F_matrix(self) -> Qobj:
        pass

    @abstractmethod
    def elementary_braid(self, braid_index: int) -> Qobj:
        pass
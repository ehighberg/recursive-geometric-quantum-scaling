from qutip import Qobj
from simulations.scripts.braid_generators.braid_generator_factory import BraidGeneratorFactory
import numpy as np

class TopologicalComposer:
    """
    Handles the composition of braiding operations for topological anyons.
    """
    def __init__(self, dimensionality: int):
        self.braid_generator = BraidGeneratorFactory.get_braid_generator(dimensionality)

    def compose_braids(self, braid_indices: list) -> Qobj:
        """
        Composes a sequence of braiding operations based on the provided indices.

        Parameters:
        - braid_indices (list): A list of integers representing braid operations.

        Returns:
        - Qobj: The resulting composed braiding operation.
        """
        composed = Qobj(np.identity(self.braid_generator.F_matrix().shape[0]))
        for index in braid_indices:
            composed = self.braid_generator.elementary_braid(index) @ composed
        return composed
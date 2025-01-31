from qutip import Qobj
from simulations.scripts.braid_generators.braid_generator_factory import BraidGeneratorFactory
import numpy as np

class TopologicalComposer:
    """
    Composes sequences of braiding operations for topological anyons.
    
    Given a list of braid indices, this class obtains the corresponding elementary 
    braid operators and composes them (via matrix multiplication) to yield a single 
    unitary operation representing the overall braiding.
    
    Attributes:
        braid_generator: An instance of a braid generator (e.g., for 2D Fibonacci anyons).
    """
    def __init__(self, dimensionality: int):
        self.braid_generator = BraidGeneratorFactory.get_braid_generator(dimensionality)

    def compose_braids(self, braid_indices: list) -> Qobj:
        """
        Compose a sequence of braiding operations based on the provided indices.
        
        Parameters:
            braid_indices (list): A list of integers (e.g., [1, 2, 1]) representing the sequence.
            
        Returns:
            Qobj: The resulting unitary operator from the composition of the braids.
        """
        # Start with the identity operator of appropriate dimension.
        composed = Qobj(np.identity(self.braid_generator.F_matrix().shape[0]))
        for index in braid_indices:
            # Multiply braids sequentially (order matters).
            composed = self.braid_generator.elementary_braid(index) @ composed
        return composed

"""
Factory for creating braid generators of different dimensionalities.
"""

from .braid_generator_2d import BraidGenerator2D

class BraidGeneratorFactory:
    """Factory class for creating appropriate braid generators."""
    
    @staticmethod
    def get_braid_generator(dimensionality: int):
        """
        Create and return a braid generator for the specified dimensionality.
        
        Args:
            dimensionality (int): The dimensionality of the Hilbert space.
            
        Returns:
            AbstractBraidGenerator: A braid generator instance.
            
        Raises:
            ValueError: If the requested dimensionality is not supported.
        """
        if dimensionality == 2:
            return BraidGenerator2D()
        else:
            raise ValueError(f"Unsupported dimensionality: {dimensionality}. Currently only 2D is supported.")

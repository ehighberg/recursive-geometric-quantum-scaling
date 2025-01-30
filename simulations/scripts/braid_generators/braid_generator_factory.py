from .braid_generator_2d import BraidGenerator2D
# Future implementations can be added here, e.g., BraidGenerator3D

class BraidGeneratorFactory:
    @staticmethod
    def get_braid_generator(dimensionality: int):
        if dimensionality == 2:
            return BraidGenerator2D(dimensionality)
        # elif dimensionality == 3:
        #     return BraidGenerator3D(dimensionality)
        else:
            raise ValueError(f"Unsupported dimensionality: {dimensionality}")
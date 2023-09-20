from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class NormalBias(Initializer):
    def __call__(self, output_shape: int) -> np.ndarray:
        return np.random.normal(0, 1, (1, output_shape))

from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class RandomBias(Initializer):
    def __call__(self, output_shape: int) -> np.ndarray:
        return np.random.rand(1, output_shape)

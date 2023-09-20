from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class RandomWeight(Initializer):
    def __call__(self, input_shape: int, output_shape: int) -> np.ndarray:
        return np.random.rand(input_shape, output_shape)

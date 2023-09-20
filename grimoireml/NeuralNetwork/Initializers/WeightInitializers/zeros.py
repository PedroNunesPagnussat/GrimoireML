from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class ZerosWeight(Initializer):
    def __call__(self, input_shape: int, output_shape: int) -> np.ndarray:
        return np.zeros((input_shape, output_shape))

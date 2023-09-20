from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class XavierUniformWeight(Initializer):
    def __call__(self, input_shape: int, output_shape: int) -> np.ndarray:
        return np.random.uniform(
            -np.sqrt(6 / (input_shape + output_shape)),
            np.sqrt(6 / (input_shape + output_shape)),
            (input_shape, output_shape),
        )

from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class XavierNormalWeight(Initializer):        
    def __call__(self, input_shape: int, output_shape: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(1 / input_shape + output_shape), (input_shape, output_shape))
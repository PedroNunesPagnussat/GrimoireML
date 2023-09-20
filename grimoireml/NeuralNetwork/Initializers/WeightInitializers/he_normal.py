from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class HeNormalWeight(Initializer):        
    def __call__(self, input_shape: int, output_shape: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(2 / input_shape), (input_shape, output_shape))
    
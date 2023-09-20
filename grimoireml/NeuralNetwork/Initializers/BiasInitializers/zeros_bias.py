from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class ZerosBias(Initializer):        
    def __call__(self, output_shape: int) -> np.ndarray:
        return np.zeros((1, output_shape), dtype=np.float64)

    
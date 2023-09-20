from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class UniformBias(Initializer):        
    def __call__(self, output_shape: int) -> np.ndarray:
        return np.random.uniform(-1, 1, (1, output_shape))
from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
import numpy as np


class ConstantBias(Initializer):    
    def __init__(self, constant: float):
        self.constant = constant

    def __call__(self, output_shape: int) -> np.ndarray:
        return np.ones((1, output_shape)) * self.constant
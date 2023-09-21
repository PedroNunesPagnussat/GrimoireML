from grimoireml.NeuralNetwork.Layers.activation import Activation
import numpy as np


class LeakyReLU(Activation):
    """This is the LeakyReLU activation layer"""

    def __init__(self, alpha=0.01):
        """This is the constructor for the ReLU class"""
        self.alpha = alpha

        def leakyrelu(x):
            return np.where(x > 0, x, self.alpha * x)

        def leakyrelu_derivative(x):
            return np.where(x > 0, 1, self.alpha)

        super().__init__(leakyrelu, leakyrelu_derivative)

    def __str__(self):
        return f"LeakyReLU(Alpha={self.alpha})"

from grimoireml.NeuralNetwork.Layers.activation import Activation
import numpy as np


class Linear(Activation):
    """This is the Linear activation layer"""

    def __init__(self):
        """This is the constructor for the Linear class"""

        def linear(x):
            return x

        def linear_derivative(x):
            return np.ones(x.shape)

        super().__init__(linear, linear_derivative)

    def __str__(self):
        return "Linear"

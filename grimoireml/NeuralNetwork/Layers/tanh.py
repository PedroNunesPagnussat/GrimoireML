from grimoireml.NeuralNetwork.Layers.activation import Activation
import numpy as np


class Tanh(Activation):
    """This is the Tanh activation layer"""

    def __init__(self):
        """This is the constructor for the Tanh class"""

        def tanh(x):
            return np.tanh(x)

        def tanh_derivative(x):
            return 1 - np.square(np.tanh(x))

        super().__init__(tanh, tanh_derivative)

    def __str__(self):
        return "Tanh"

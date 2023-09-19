from grimoireml.NeuralNetwork.Layers.activation import Activation
import numpy as np

class ReLU(Activation):
    """This is the ReLU activation layer"""

    def __init__(self):
        """This is the constructor for the ReLU class"""
        def relu(x):
            return np.maximum(0, x)
        
        def relu_derivative(x):
            np.where(x > 0, 1, 0)

        super().__init__(relu, relu_derivative)

    def __str__(self):
        return "ReLU"
from grimoireml.NeuralNetwork.Layers.layer import Layer
import numpy as np


class Activation(Layer):
    """This is the template for the activation layer"""

    def __init__(self, activation, derivative):
        """This is the constructor for the Activation class"""
        self._activation = activation
        self._derivative = derivative
        self.trainable = False

    def __call__(self, input_layer: Layer) -> Layer:
        """This method will be called when the object is called"""
        self.input_shape = input_layer.output_shape
        self.output_shape = input_layer.output_shape
        return self

    def _forward(self, input_data: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        self._input_data = input_data
        return self._activation(input_data)

    def _backward(self, accumulated_error: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        return np.multiply(accumulated_error, self._derivative(self._input_data))

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        return self._activation(input_data)

    def __str__(self):
        pass

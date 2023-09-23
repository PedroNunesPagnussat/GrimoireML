import numpy as np
from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Initializers.initializer import Initializer
from grimoireml.NeuralNetwork.Initializers.WeightInitializers.xavier_uniform import (
    XavierUniformWeight,
)
from grimoireml.NeuralNetwork.Initializers.WeightInitializers.he_uniform import (
    HeUniformWeight,
)
from grimoireml.NeuralNetwork.Initializers.BiasInitializers import ZerosBias


class Dense(Layer):
    """This is the main layer, the fully connected dense layer"""

    def __init__(
        self,
        output_shape: int,
        input_shape: tuple = None,
        weight_initializer: Initializer = None,
        bias_initializer: Initializer = None,
    ):
        """This is the constructor for the Dense class"""

        if weight_initializer is None:
            self.weight_initializer = HeUniformWeight()
        else:
            self.weight_initializer = weight_initializer
        if bias_initializer is None:
            self.bias_initializer = ZerosBias()
        else:
            self.bias_initializer = bias_initializer

        super().__init__(output_shape)

        if input_shape is not None:
            self.input_shape = input_shape[0]
            self.weights = self.weight_initializer(input_shape[0], output_shape)
            self.bias = self.bias_initializer(output_shape)

        else:
            self.weights = None
            self.bias = None

        self.bias_gradient = None
        self.weights_gradient = None
        self.input_data = None
        self.trainable = True

    def __call__(self, input_layer: Layer) -> Layer:
        """This is the representation of the call method"""
        self.input_shape = input_layer.output_shape
        self.weights = self.weight_initializer(
            input_layer.output_shape, self.output_shape
        )
        self.bias = self.bias_initializer(self.output_shape)
        return self

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """This is the representation of the forward pass"""
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, accumulated_error: np.ndarray) -> np.ndarray:
        """This is the representation of the backward pass"""
        propagate_error = np.dot(accumulated_error, self.weights.T)
        self.weights_gradient = np.dot(self.input_data.T, accumulated_error)
        self.bias_gradient = np.sum(accumulated_error)
        return propagate_error

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """This is the representation of the predict method"""
        return self.forward(input_data)

    def __str__(self) -> str:
        return (
            f"Dense Layer with {self.input_shape} inputs, "
            f"{self.output_shape} outputs, and "
            f"{self.weights.size + self.bias.size} parameters"
        )

import numpy as np
from grimoireml.NeuralNetwork.Layers.layer import Layer


class Dense(Layer):
    """This is the main layer, the fully connected dense layer"""

    def __init__(self,
                 output_shape: int, input_shape: tuple = None,
                 weight_initializer: callable = None,
                 bias_initializer: callable = None):
        """This is the constructor for the Dense class"""

        if weight_initializer is None:
            weight_initializer = np.random.randn
        if bias_initializer is None:
            bias_initializer = np.zeros

        super().__init__(output_shape)

        if input_shape is not None:
            self.input_shape = input_shape[0]
            self.weights = weight_initializer(input_shape[0], output_shape)
            self.bias = bias_initializer((1, output_shape))

        else:
            self.weights = None
            self.bias = None

        self._bias_gradient = None
        self._weight_gradient = None
        self._input_data = None

    def __call__(self, input_layer: Layer) -> Layer:
        """This is the representation of the call method"""
        self.input_shape = input_layer.output_shape
        self.weights = np.random.randn(input_layer.output_shape, self.output_shape)
        self.bias = np.zeros((1, self.output_shape))

        return self

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """This is the representation of the forward pass"""
        self._input_data = input_data
        return np.dot(input_data, self.weights) + self.bias

    def backward(self, accumulated_error: np.ndarray) -> np.ndarray:
        """This is the representation of the backward pass"""
        propagate_error = np.dot(self.weights.T, accumulated_error)
        self._weight_gradient = np.dot(accumulated_error, self._input_data.T)
        self._bias_gradient = accumulated_error
        return propagate_error

    def __str__(self) -> str:
        return (f"Dense Layer with {self.input_shape} inputs, "
                f"{self.output_shape} outputs, and {self.weights.size + self.bias.size} parameters")

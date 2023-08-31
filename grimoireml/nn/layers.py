from abc import ABC, abstractmethod
import numpy as np
from ..functions import activation_functions

class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Dense(Layer):
    def __init__(self, neurons, activation):
        """Initialize a dense layer with neurons and activation function."""
        self._neurons = neurons
        self._activation, self._activation_derivative = activation_functions.get_activation_function(activation)
        self._weights = None
        self._biases = None
        self._sum = None
        self._output = None
        self._delta = None

    def  _initialize_weights_and_bias(self, input_shape):
        """Initialize weights and biases for the layer."""
        self._weights = np.random.uniform(-1, 1, size=(input_shape, self._neurons))
        self._biases = np.random.uniform(-1, 1, size=(self._neurons,))

    def __str__(self):
        """Return string representation of the dense layer."""
        return f"Dense layer with {self._neurons} neurons and {self._activation} activation"


class Input(Layer):
    def __init__(self, input_shape):
        """Initialize an input layer with a given shape."""
        self._input_shape = input_shape


    def __str__(self):
        """Return string representation of the input layer."""
        return f"Input layer with shape {self._input_shape}"

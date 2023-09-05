from abc import ABC, abstractmethod
from typing import Tuple, Callable, Union
import numpy as np
from ..functions import activation_functions
from .weight_initializers import get_initializer

class Layer(ABC):
    """
    Abstract base class for different types of layers.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the layer."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the layer."""
        pass


class Dense(Layer):
    """
    Class representing a dense layer in a neural network.
    
    Attributes:
        _neurons (int): Number of neurons in the layer.
        _activation (Callable): Activation function.
        _activation_derivative (Callable): Derivative of the activation function.
        _weights (np.ndarray): Weights of the layer.
        _biases (np.ndarray): Biases of the layer.
        _sum (np.ndarray): Weighted sum before activation.
        _output (np.ndarray): Output after activation.
        _delta (np.ndarray): Error term for backpropagation.
        _weight_grad (np.ndarray): The gradient for the weigths.
        _bias_grad (np.ndarray): The gradient for the Bias.
        
    """
    
    def __init__(self, neurons: int, activation: str, initialaizer: str = "glorot_uniform", active_bias: bool = True):
        """
        Initialize a dense layer with neurons and activation function.
        
        Args:
            neurons (int): Number of neurons in the layer.
            activation (str): Name of the activation function.
        """
        self._neurons = neurons
        self._activation, self._activation_derivative = activation_functions.get_activation_function(activation)
        self._weights = None
        self._biases = None
        self._sum = None
        self._output = None
        self._delta = None
        self._weights_grad = None
        self._bias_grad = None
        self._initializer = initialaizer

    def _initialize_weights_and_bias(self, input_shape: int):
        """
        Initialize weights and biases for the layer.
        
        Args:
            input_shape (int): The shape of the input data.
        """
        initializer = get_initializer(self._initializer)
        self._weights = initializer(input_shape, self._neurons)
        self._biases = np.random.uniform(-1, 1, size=(self._neurons,))


    def __str__(self) -> str:
        """
        Return string representation of the dense layer.
        
        Returns:
            str: Description of the dense layer.
        """
        return f"Dense layer with {self._neurons} neurons and {self._activation.__name__} activation"


class Input(Layer):
    """
    Class representing an input layer in a neural network.
    
    Attributes:
        _input_shape (Tuple[int, int]): Shape of the input data.
    """
    
    def __init__(self, input_shape: Tuple[int, int]):
        """
        Initialize an input layer with a given shape.
        
        Args:
            input_shape (Tuple[int, int]): Shape of the input data.
        """
        self._input_shape = input_shape

    def __str__(self) -> str:
        """
        Return string representation of the input layer.
        
        Returns:
            str: Description of the input layer.
        """
        return f"Input layer with shape {self._input_shape}"

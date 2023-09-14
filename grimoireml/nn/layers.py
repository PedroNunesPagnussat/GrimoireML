from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from .initializers.weight_initializers import WeightInitializer, GlorotUniformWeightInitializer
from .initializers.bias_initializers import BiasInitializer, ZerosBiasInitializer
from .functions.activation_functions import ActivationFunction

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

    @abstractmethod
    def _forward(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Output of the layer.
        """
        pass

    @abstractmethod
    def _backward(self, error: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.
        
        Args:
            error (np.ndarray): Error term.
            
        Returns:
            np.ndarray: Error term for the previous layer.
        """
        pass

    @abstractmethod
    def _compute_gradients(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients for the layer.
        
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients for the weights and bias.
        """
        pass


    @abstractmethod
    def _predict(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass, this is not important in the .
        
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Output of the layer.
        """
        pass

class Dense(Layer):
    """
    Class representing a dense layer in a neural network.
    
    Attributes:
        _neurons (int): Number of neurons in the layer.
        _activation (Callable): Activation function.
        _activation_derivative (Callable): Derivative of the activation function.
        _weights (np.ndarray): Weights of the layer.
        _bias (np.ndarray): Bias of the layer.
        _sum (np.ndarray): Weighted sum before activation.
        _output (np.ndarray): Output after activation.
        _delta (np.ndarray): Error term for backpropagation.
        _weight_grad (np.ndarray): The gradient for the weigths.
        _bias_grad (np.ndarray): The gradient for the Bias.
        
    """
    
    def __init__(
                    self, input_shape: int, neurons: int, 
                    activation: ActivationFunction, 
                    weight_initializer: WeightInitializer = GlorotUniformWeightInitializer(), bias_initializer: BiasInitializer = ZerosBiasInitializer()
                ):
        """
        Initialize a dense layer with neurons and activation function.
        
        Args:
            neurons (int): Number of neurons in the layer.
            activation (str): Name of the activation function.
        """

        self._input_shape = input_shape
        self._neurons = neurons
        

        self._bias = bias_initializer._initialize(neurons)
        self._weights = weight_initializer._initialize(input_shape=input_shape, output_shape=neurons)
        
        self._activation = activation
        
        self._sum = None
        self._output = None
        self._delta = None
        self._weights_grad = None
        self._bias_grad = None

    def _forward(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Output of the layer.
        """
        self._sum = np.dot(input, self._weights) + self._bias
        self._output = self._activation._compute(self._sum)
        return self._output
    
    def _predict(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass, this is not important in this layer.
    
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            np.ndarray: Output of the layer.
        """
        self._sum = np.dot(input, self._weights) + self._bias
        self._output = self._activation._compute(self._sum)
        return self._output
    

    def _backward(self, error: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.
        
        Args:
            error (np.ndarray): Error term.
            
        Returns:
            np.ndarray: Error term for the previous layer.
        """
        self._delta = error * self._activation._compute_derivative(self._sum)
        
        return np.dot(self._delta, self._weights.T)
    
    
    def _compute_gradients(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients for the layer.
        
        Args:
            input (np.ndarray): Input data.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients for the weights and bias.
        """
        self._weights_grad = np.dot(input.T, self._delta)
        self._bias_grad = np.sum(self._delta, axis=0)
        return self._weights_grad, self._bias_grad

    def __str__(self) -> str:
        """
        Return string representation of the dense layer.
        
        Returns:
            str: Description of the dense layer.
        """
        return f"Dense layer with Input_Shape: {self._input_shape}, Neurons: {self._neurons}, Activation Function: {self._activation} activation, and Trainable Parameters: {self._weights.size + self._bias.size}"




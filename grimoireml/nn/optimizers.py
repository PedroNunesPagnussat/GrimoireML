from abc import ABC, abstractmethod
from typing import List
import numpy as np

from grimoireml.nn.layers import Layer
from .layers import Layer  # Assuming layers.py is in the same directory

class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    
    Attributes:
        _lr (float): The learning rate for the optimizer.
    """
    
    def __init__(self, lr: float = 0.01):
        """
        Initialize the optimizer with a learning rate.
        
        Args:
            lr (float): The learning rate.
        """
        self._lr = lr

    def _update(self, layers: List[Layer]) -> None:
        """
        Update the weights and biases for each layer in the list.
        
        Args:
            layers (List[Layer]): List of layers to be updated.
        """
        for layer in layers:
            self._layer_update(layer)


    @abstractmethod
    def _layer_update(self, layer: Layer) -> None:
        """
        Abstract method to update the weights and biases for a single layer.
        
        Args:
            layer (Layer): The layer to be updated.
        """
        pass


    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the optimizer."""
        pass

    def _initialize_momentum(self, layer: Layer) -> None:
        """
        Initialize the squared gradient accumulators for the weights and biases of a layer.
        
        Args:
            layer (Layer): The layer for which the squared gradient accumulators are to be initialized.
        """
        layer._weights_momentum = np.zeros_like(layer._weights)
        layer._bias_momentum = np.zeros_like(layer._bias)



class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Inherits from:
        Optimizer: The abstract base class for optimizers.
    """
    
    def __init__(self, lr: float = 0.01):
        """
        Initialize the SGD optimizer with a learning rate.
        
        Args:
            lr (float): The learning rate.
        """
        super().__init__(lr)

    def _layer_update(self, layer: Layer) -> None:
        """
        Update the weights and biases for a single layer using SGD.
        
        Args:
            layer (Layer): The layer to be updated.
        """

        layer._weights -= self._lr * layer._weights_grad
        layer._bias -= self._lr * layer._bias_grad


    def __str__(self) -> str:
        """Return the string representation of the optimizer."""
        return f"SGD(lr={self._lr})"


class Adagrad(Optimizer):
    """
    Adagrad optimizer for updating model parameters.
    
    Adagrad adapts the learning rate during training and is well-suited for dealing with sparse data.
    However, its learning rate may become too small over time, effectively stopping the model from learning.
    
    Attributes:
        _lr (float): The initial learning rate.
        _epsilon (float): A small constant to avoid division by zero in learning rate adaptation.
    """
    
    def __init__(self, lr: float = 0.01):
        """
        Initialize the Adagrad optimizer.
        
        Args:
            lr (float): The initial learning rate.
        """
        super().__init__(lr)
        self._epsilon = 1e-8  # To prevent division by zero

    def _layer_update(self, layer: Layer) -> None:
        """
        Update the weights and biases for a single layer using the Adagrad algorithm.
        
        Args:
            layer (Layer): The layer whose parameters are to be updated.
        """
        # Initialize squared gradient accumulators if they don't exist
        if not hasattr(layer, "_weights_momentum"):
            self._initialize_momentum(layer)

        # Update squared gradient accumulators
        layer._weights_momentum += np.square(layer._weights_grad)
        layer._bias_momentum += np.square(layer._bias_grad)

        # Update weights and biases
        layer._weights -= self._lr / np.sqrt(layer._weights_momentum + self._epsilon) * layer._weights_grad
        layer._bias -= self._lr / np.sqrt(layer._bias_momentum + self._epsilon) * layer._bias_grad

    def __str__(self) -> str:
        """
        Return the string representation of the optimizer.
        
        Returns:
            str: A string that shows the optimizer's name and its current learning rate.
        """
        return f"Adagrad(lr={self._lr})"
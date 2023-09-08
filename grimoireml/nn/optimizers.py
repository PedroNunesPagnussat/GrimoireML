from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .layers import Layer  # Assuming layers.py is in the same directory

class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    
    Attributes:
        _lr (float): The learning rate for the optimizer.
    """
    
    def __init__(self, lr: float):
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


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Inherits from:
        Optimizer: The abstract base class for optimizers.
    """
    
    def __init__(self, lr: float):
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

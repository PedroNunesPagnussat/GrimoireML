from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class WeightInitializer(ABC):
    """
    Abstract Base Class (ABC) for weight initializers in neural networks.
    
    This class serves as an interface for all weight initializer subclasses, 
    ensuring they implement an `initialize` method.
    
    Attributes:
        None
    """

    @abstractmethod
    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix based on the given input and output shapes.

        Args:
            input_shape (int): The number of input units for which weights need to be initialized.
            output_shape (int): The number of output units for which weights need to be initialized.

        Returns:
            np.ndarray: An initialized weight matrix of shape (input_shape, output_shape).

        Raises:
            NotImplementedError: This method is meant to be overridden by subclasses.
        """
        pass


class ZeroWeightInitializer(WeightInitializer):
    """
    Initialize weights as zeros.
    
    This class generates a weight matrix filled with zeros.

    Attributes:
        None
    """

    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix filled with zeros.

        Args:
            input_shape (int): The number of input units for which weights need to be initialized.
            output_shape (int): The number of output units for which weights need to be initialized.

        Returns:
            np.ndarray: A weight matrix of zeros with shape (input_shape, output_shape).
        """
        return np.zeros((input_shape, output_shape))


class HeWeightInitializer(WeightInitializer):
    """
    Initialize weights using He initialization. Suitable for ReLU activations.

    Attributes:
        None
    """
    
    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix using He initialization.

        Args:
            input_shape (int): The number of input units for which weights need to be initialized.
            output_shape (int): The number of output units for which weights need to be initialized.

        Returns:
            np.ndarray: A weight matrix initialized using He initialization of shape (input_shape, output_shape).
        """
        return np.random.randn(input_shape, output_shape) * np.sqrt(2. / input_shape)
    

class XavierWeightInitializer(WeightInitializer):
    """
    Initialize weights using Xavier initialization. Suitable for tanh activations.
    
    Attributes:
        None
    """

    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix using Xavier initialization.

        Args:
            input_shape (int): The number of input units for the layer.
            output_shape (int): The number of output units for the layer.

        Returns:
            np.ndarray: A weight matrix initialized using Xavier initialization of shape (input_shape, output_shape).
        """
        return np.random.randn(input_shape, output_shape) * np.sqrt(1. / input_shape)


class UniformWeightInitializer(WeightInitializer):
    """
    Initialize weights with random values sampled uniformly.
    
    Attributes:
        None
    """

    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix with random values sampled uniformly.

        Args:
            input_shape (int): The number of input units for the layer.
            output_shape (int): The number of output units for the layer.

        Returns:
            np.ndarray: A weight matrix filled with random values of shape (input_shape, output_shape).
        """
        return np.random.uniform(-1, 1, (input_shape, output_shape))


class NormalWeightInitializer(WeightInitializer):
    """
    Initialize weights with random values sampled from a normal distribution.
    
    Attributes:
        None
    """

    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix with random values sampled from a normal distribution.

        Args:
            input_shape (int): The number of input units for the layer.
            output_shape (int): The number of output units for the layer.

        Returns:
            np.ndarray: A weight matrix filled with random values of shape (input_shape, output_shape).
        """
        return np.random.randn(input_shape, output_shape)


class GlorotUniformWeightInitializer(WeightInitializer):
    """
    Initialize weights using Glorot uniform initialization, also known as Xavier uniform initialization.
    
    Attributes:
        None
    """

    def _initialize(self, input_shape: int, output_shape: int) -> np.ndarray:
        """
        Initialize a weight matrix using Glorot uniform initialization.

        Args:
            input_shape (int): The number of input units for the layer.
            output_shape (int): The number of output units for the layer.

        Returns:
            np.ndarray: A weight matrix initialized using Glorot uniform initialization of shape (input_shape, output_shape).
        """
        limit = np.sqrt(6 / (input_shape + output_shape))
        return np.random.uniform(-limit, limit, (input_shape, output_shape))
    
    def __str__(self) -> str:
        """Return the string representation of the optimizer."""
        return "Glorot Uniform"
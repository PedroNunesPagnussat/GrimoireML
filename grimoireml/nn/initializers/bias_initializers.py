from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

class BiasInitializer(ABC):
    """
    Abstract Base Class (ABC) for bias initializers in neural networks.
    
    This class serves as an interface for all bias initializer subclasses, 
    ensuring they implement an `initialize` method.
    
    Attributes:
        None
    """

    @abstractmethod
    def _initialize(self, output_shape: int) -> np.ndarray:
        """
        Initialize a bias vector based on the given shape.

        Args:
            output_shape (int): The number of output units for which biases need to be initialized.

        Returns:
            np.ndarray: An initialized bias vector of shape (output_shape,).

        Raises:
            NotImplementedError: This method is meant to be overridden by subclasses.
        """
        pass


class ZerosBiasInitializer(BiasInitializer):
    """
    Initialize biases as zeros.
    
    This class generates a bias vector filled with zeros.

    Attributes:
        None
    """
    def _initialize(self, output_shape: int) -> np.ndarray:
        """
        Initialize a bias vector filled with zeros.

        Args:
            output_shape (int): The number of output units for which biases need to be initialized.

        Returns:
            np.ndarray: A bias vector of zeros with shape (output_shape,).
        """
        return np.zeros((output_shape,))


class OnesBiasInitializer(BiasInitializer):
    """
    Initialize biases as ones.
    
    This class generates a bias vector filled with ones.

    Attributes:
        None
    """
    def _initialize(self, output_shape: int) -> np.ndarray:
        """
        Initialize a bias vector filled with ones.

        Args:
            output_shape (int): The number of output units for which biases need to be initialized.

        Returns:
            np.ndarray: A bias vector of ones with shape (output_shape,).
        """
        return np.ones((output_shape,))


class RandomBiasInitializer(BiasInitializer):
    """
    Initialize biases with random values.
    
    This class generates a bias vector filled with random values sampled from a uniform distribution
    between -1 and 1.

    Attributes:
        None
    """
    def _initialize(self, output_shape: int) -> np.ndarray:
        """
        Initialize a bias vector filled with random values.

        Args:
            output_shape (int): The number of output units for which biases need to be initialized.

        Returns:
            np.ndarray: A bias vector of random values with shape (output_shape,).
        """
        return np.random.uniform(-1, 1, (output_shape,))


class SmallConstantBiasInitializer(BiasInitializer):
    """
    Initialize biases with a small constant value.
    
    This class generates a bias vector filled with a small constant value. 
    The constant value is a parameter that can be set during initialization of this class.

    Attributes:
        constant (float): The constant value to initialize the bias vector with.
    """
    def __init__(self, constant: float = 0.01):
        """
        Constructor for SmallConstantBiasInitializer class.

        Args:
            constant (float, optional): The small constant value to initialize the bias vector with.
                                        Default is 0.01.
        """
        self.constant = constant

    def _initialize(self, output_shape: int) -> np.ndarray:
        """
        Initialize a bias vector filled with a small constant value.

        Args:
            output_shape (int): The number of output units for which biases need to be initialized.

        Returns:
            np.ndarray: A bias vector filled with the small constant value and with shape (output_shape,).
        """
        return np.full((output_shape,), self.constant)

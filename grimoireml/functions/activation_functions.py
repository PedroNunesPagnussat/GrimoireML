from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    """
    @abstractmethod
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the activation function.
        """
        pass

    @abstractmethod
    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the activation function.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        pass

# Tip: Consistent naming and documentation make the code more maintainable.
class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the sigmoid activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the sigmoid activation function.
        """
        sx = self._activation(x)
        return sx * (1 - sx)

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return "Sigmoid"

# Tip: Consider using NumPy's built-in functions for better performance.
class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the ReLU activation function.
        """
        return np.maximum(x, 0, out=x)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the ReLU activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the ReLU activation function.
        """
        return np.where(x > 0, 1, 0)

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return "ReLU"

# Tip: Use descriptive comments to explain non-obvious code.
class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent (Tanh) activation function.
    """
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Tanh activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the Tanh activation function.
        """
        return np.tanh(x)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Tanh activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the Tanh activation function.
        """
        return 1 - np.tanh(x) ** 2

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return "Tanh"

# Tip: Be cautious with numerical stability, especially with functions like Softmax.
class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Softmax activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the Softmax activation function.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Softmax activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the Softmax activation function.
        """
        s = self._activation(x)
        return s * (1 - s)

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return "Softmax"


# Tip: Keep the interface consistent across different activation functions for easier usage and maintenance.
class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the linear activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the linear activation function.
        """
        return x

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the linear activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the linear activation function.
        """
        return np.ones_like(x)

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return "Linear"

# Tip: When implementing variants, consider parameterizing them for greater flexibility.
class LeakyReLU(ActivationFunction):
    """
    Leaky Rectified Linear Unit (LeakyReLU) activation function.
    """
    def __init__(self, alpha: float = 0.01):
        """
        Initialize the LeakyReLU activation function.

        Args:
            alpha (float): The slope for negative values. Default is 0.01.
        """
        self.alpha = alpha

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the LeakyReLU activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the LeakyReLU activation function.
        """
        return np.where(x > 0, x, self.alpha * x)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the LeakyReLU activation function to the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The array after applying the derivative of the LeakyReLU activation function.
        """
        return np.where(x > 0, 1, self.alpha)

    def __str__(self) -> str:
        """
        Return the name of the activation function as a string.

        Returns:
            str: The name of the activation function.
        """
        return f"LeakyReLU(alpha={self.alpha})"
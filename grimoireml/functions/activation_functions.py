from typing import Callable, Tuple
import numpy as np


def get_activation_function(activation: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Return activation function and its derivative based on the input string.
    
    Args:
        activation (str): The name of the activation function ("sigmoid", "relu", "tanh", "softmax", "leaky_relu", "elu", "selu", "linear").
        
    Returns:
        Tuple[Callable, Callable]: The activation function and its derivative.
        
    Raises:
        ValueError: If the activation function specified is not supported.
    """
    if activation not in activation_map:
        raise ValueError(f"Activation function {activation} not supported. Supported activations are: {sorted(list(activation_map.keys()))}")
    return activation_map[activation]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of sigmoid function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the sigmoid function.
    """
    sx = _sigmoid(x)
    return sx * (1 - sx)

def _relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the ReLU function.
    """
    return np.maximum(x, 0, out=x)

def _relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of ReLU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)

def _tanh(x: np.ndarray) -> np.ndarray:
    """
    Apply tanh function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the tanh function.
    """
    return np.tanh(x)

def _tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of tanh function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the tanh function.
    """
    return 1 - np.tanh(x) ** 2

def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Apply softmax function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the softmax function.
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def _softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of softmax function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the softmax function.
    """
    s = _softmax(x)
    return s * (1 - s)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Apply Leaky ReLU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        alpha (float): The slope for negative values. Default is 0.01.
        
    Returns:
        np.ndarray: The array after applying the Leaky ReLU function.
    """
    return np.where(x > 0, x, alpha * x)

def _leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Apply derivative of Leaky ReLU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        alpha (float): The slope for negative values. Default is 0.01.
        
    Returns:
        np.ndarray: The array after applying the derivative of the Leaky ReLU function.
    """
    return np.where(x > 0, 1, alpha)

def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Apply ELU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        alpha (float): The alpha value for ELU. Default is 1.0.
        
    Returns:
        np.ndarray: The array after applying the ELU function.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def _elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Apply derivative of ELU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        alpha (float): The alpha value for ELU. Default is 1.0.
        
    Returns:
        np.ndarray: The array after applying the derivative of the ELU function.
    """
    return np.where(x > 0, 1, alpha * np.exp(x))

def _selu(x: np.ndarray) -> np.ndarray:
    """
    Apply SELU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the SELU function.
    """
    alpha, scale = 1.67326, 1.0507
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def _selu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of SELU function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the SELU function.
    """
    alpha, scale = 1.67326, 1.0507
    return scale * np.where(x > 0, 1, alpha * np.exp(x))

def _linear(x: np.ndarray) -> np.ndarray:
    """
    Apply Linear function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the Linear function.
    """
    return x

def _linear_derivative(x: np.ndarray) -> np.ndarray:
    """
    Apply derivative of Linear function to input array.
    
    Args:
        x (np.ndarray): The input array.
        
    Returns:
        np.ndarray: The array after applying the derivative of the Linear function.
    """
    return np.ones_like(x)


activation_map = {
    "sigmoid" : (_sigmoid, _sigmoid_derivative),
    "relu" : (_relu, _relu_derivative),
    "tanh": (_tanh, _tanh_derivative),
    "softmax" : (_softmax, _softmax_derivative),
    "leaky_relu": (_leaky_relu, _leaky_relu_derivative),
    "elu": (_elu, _elu_derivative),
    "selu": (_selu, _selu_derivative),
    "linear": (_linear, _linear_derivative)
}

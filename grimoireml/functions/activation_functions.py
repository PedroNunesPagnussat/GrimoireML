from typing import Callable, Tuple
import numpy as np

def get_activation_function(activation: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Return activation function and its derivative based on input string.
    
    Args:
        activation (str): The name of the activation function ("sigmoid", "relu", "tanh", "softmax").
        
    Returns:
        Tuple[Callable, Callable]: The activation function and its derivative.
        
    Raises:
        Exception: If an invalid activation function name is provided.
    """
    if activation == "sigmoid":
        return _sigmoid, _sigmoid_derivative
    elif activation == "relu":
        return _relu, _relu_derivative
    elif activation == "tanh":
        return _tanh, _tanh_derivative
    elif activation == "softmax":
        return _softmax, _softmax_derivative
    raise Exception("Invalid activation function")

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

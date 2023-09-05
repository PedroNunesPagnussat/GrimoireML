import numpy as np
from typing import Callable

def _zeros_bias_initializer(output_shape: int) -> np.ndarray:
    """
    Initialize a bias vector with zeros.
    
    Args:
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A bias vector filled with zeros.
    """
    return np.zeros((output_shape,))

def _ones_bias_initializer(output_shape: int) -> np.ndarray:
    """
    Initialize a bias vector with ones.
    
    Args:
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A bias vector filled with ones.
    """
    return np.ones((output_shape,))

def _random_bias_initializer(output_shape: int) -> np.ndarray:
    """
    Initialize a bias vector with random values sampled uniformly.
    
    Args:
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A bias vector filled with random values sampled uniformly.
    """
    return np.random.uniform(-1, 1, (output_shape,))

# Map the bias initializers to their corresponding functions


def _small_constant_bias_initializer(output_shape: int, constant: float = 0.01) -> np.ndarray:
    """
    Initialize a bias vector with a small constant value.
    
    Args:
        output_shape (int): Number of output units for the layer.
        constant (float): The small constant value to initialize the bias vector with.
        
    Returns:
        np.ndarray: A bias vector filled with the small constant value.
    """
    return np.full((output_shape,), constant)

# Update the bias initializers map
_bias_initializers_map = {
    'Zeros': _zeros_bias_initializer,
    'Ones': _ones_bias_initializer,
    'Random': _random_bias_initializer,
    'SmallConstant': _small_constant_bias_initializer
}

def get_bias_initializer(name: str) -> Callable[[int], np.ndarray]:
    """
    Retrieve a bias initializer function by its name.
    
    Args:
        name (str): The name of the desired bias initializer.
        
    Returns:
        function: The corresponding bias initializer function.
        
    Raises:
        ValueError: If the specified initializer name is not found in the available initializers.
        
    Example:
        >>> initializer = get_bias_initializer('Zeros')
        >>> biases = initializer(64)
    """
    if name in _bias_initializers_map:
        return _bias_initializers_map[name]
    else:
        raise ValueError(f"Initializer {name} not found. Available initializers are: {list(_bias_initializers_map.keys())}")

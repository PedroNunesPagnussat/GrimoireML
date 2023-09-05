import numpy as np

def _zeros_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix with zeros.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix filled with zeros.
    """
    return np.zeros((input_shape, output_shape))

def _he_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix using He initialization, suitable for ReLU activations.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix initialized using He initialization.
    """
    return np.random.randn(input_shape, output_shape) * np.sqrt(2. / input_shape)

def _xavier_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix using Xavier initialization, suitable for tanh activations.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix initialized using Xavier initialization.
    """
    return np.random.randn(input_shape, output_shape) * np.sqrt(1. / input_shape)

def _uniform_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix with random values sampled uniformly.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix filled with random values sampled uniformly.
    """
    return np.random.uniform(-1, 1, (input_shape, output_shape))

def _normal_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix with random values sampled from a normal distribution.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix filled with random values sampled from a normal distribution.
    """
    return np.random.randn(input_shape, output_shape)

def _glorot_uniform_initializer(input_shape, output_shape):
    """
    Initialize a weight matrix using Glorot uniform initialization, also known as Xavier uniform initialization.
    
    Args:
        input_shape (int): Number of input units for the layer.
        output_shape (int): Number of output units for the layer.
        
    Returns:
        np.ndarray: A weight matrix initialized using Glorot uniform initialization.
    """
    limit = np.sqrt(6 / (input_shape + output_shape))
    return np.random.uniform(-limit, limit, (input_shape, output_shape))

# Map the initializers to their corresponding functions
_initializers_map = {
    'zeros': _zeros_initializer,
    'He': _he_initializer,
    'Xavier': _xavier_initializer,
    'uniform': _uniform_initializer,
    'normal': _normal_initializer,
    'glorot_uniform': _glorot_uniform_initializer
}

def get_initializer(name):
    """
    Retrieve a weight initializer function by its name.
    
    Args:
        name (str): The name of the desired weight initializer.
        
    Returns:
        function: The corresponding weight initializer function.
        
    Raises:
        ValueError: If the specified initializer name is not found in the available initializers.
        
    Example:
        >>> initializer = get_initializer('He')
        >>> weights = initializer(256, 64)
    """
    if name in _initializers_map:
        return _initializers_map[name]
    else:
        raise ValueError(f"Initializer {name} not found. Available initializers are: {list(_initializers_map.keys())}")

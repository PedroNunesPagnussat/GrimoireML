import numpy as np

def get_activation_function(activation):
    """Return activation function and its derivative based on input string."""
    if activation == "sigmoid":
        return _sigmoid, _sigmoid_derivative
    elif activation == "relu":
        return _relu, _relu_derivative
    elif activation == "tanh":
        return _tanh, _tanh_derivative
    elif activation == "softmax":
        return _softmax, _softmax_derivative
    raise Exception("Invalid activation function")

def _sigmoid(x):
    """Apply sigmoid function to input array."""
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(x):
    """Apply derivative of sigmoid function to input array."""
    sx = _sigmoid(x)
    return sx * (1 - sx)

def _relu(x):
    """Apply ReLU function to input array."""
    return np.maximum(x, 0, out=x)

def _relu_derivative(x):
    """Apply derivative of ReLU function to input array."""
    return np.where(x > 0, 1, 0)

def _tanh(x):
    """Apply tanh function to input array."""
    return np.tanh(x)

def _tanh_derivative(x):
    """Apply derivative of tanh function to input array."""
    return 1 - np.tanh(x) ** 2

def _softmax(x):
    """Apply softmax function to input array."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def _softmax_derivative(x):
    """Apply derivative of softmax function to input array."""
    s = _softmax(x)
    return s * (1 - s)

import numpy as np  

def get_activation_function(activation):
    if type(activation) == str:
        if activation == "sigmoid":
            return sigmoid, sigmoid_derivative
        elif activation == "relu":
            return relu, relu_derivative
        elif activation == "tanh":
            return tanh, tanh_derivative
        elif activation == "softmax":
            return softmax, softmax_derivative


        
    raise Exception("Invalid activation function")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

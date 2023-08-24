from abc import ABC, abstractmethod
import numpy as np
from grimoireml.functions import activation_functions
# import loss_functions


# Abstract Layer class
class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

        

# Dense Layer inheriting from Layer
class Dense(Layer):
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation, self.activation_derivative = activation_functions.get_activation_function(activation)
        self.weights = None
        self.biases = None
        self.output = None
        self.delta = None

        # self.output_derivative = None

    def initialize_weights(self, input_shape):
        self.weights = np.random.uniform(-1, 1, size=(input_shape, self.neurons))
    
    def initialize_biases(self):
        self.biases = np.random.uniform(-1, 1, size=(self.neurons,))

    def print_weights(self):
        print(f"Layer weights:\n{self.weights}")



    def __str__(self):
        return f"Dense layer with {self.neurons} neurons and {self.activation} activation"

# Input Layer inheriting from Layer

class Input(Layer):
    def __init__(self, input_shape):
        self.output = None
        self.input_shape = input_shape
        

    def __str__(self):
        return f"Input layer with shape {self.input_shape}"

    
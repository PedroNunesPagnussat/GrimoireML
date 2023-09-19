import numpy as np
from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def update_layer(self, layer: Layer) -> None:
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.bias -= self.learning_rate * layer.bias_gradient

    def __str__(self):
        return f"SGD(learning_rate={self.learning_rate})"



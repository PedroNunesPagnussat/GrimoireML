from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer
import numpy as np


class Adagrad(Optimizer):
    """This is the Adam optimizer"""

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def update_params(self, layers: np.array) -> None:
        """This updates all the layers in the model."""

        for layer in layers:
            if layer.trainable:
                if not hasattr(layer, "_m"):
                    self.initialize_adagrad_attrs(layer)
                self.update_layer(layer)

    def update_layer(self, layer: Layer) -> None:
        # Weights
        layer._m = np.square(layer.weights_gradient)
        layer._m_bias = np.square(layer.bias_gradient)

        layer.weights -= (
            self.learning_rate * layer.weights_gradient / (np.sqrt(layer._m) + 1e-8)
        )
        layer.bias -= (
            self.learning_rate * layer.bias_gradient / (np.sqrt(layer._m_bias) + 1e-8)
        )

    def initialize_adagrad_attrs(self, layer: Layer) -> None:
        layer._m = np.zeros_like(layer.weights)
        layer._m_bias = np.zeros_like(layer.bias)

    def __str__(self) -> str:
        return f"Adagrad(learning_rate={self.learning_rate})"

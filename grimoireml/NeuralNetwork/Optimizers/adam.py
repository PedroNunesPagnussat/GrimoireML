from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    """This is the Adam optimizer"""

    def __init__(
        self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_step = 0

    def update_params(self, layers: np.array) -> None:
        """This updates all the layers in the model."""
        self.time_step += 1
        for layer in layers:
            if layer.trainable:
                if not hasattr(layer, "_m"):
                    self.initialize_adam_attrs(layer)
                self.update_layer(layer)

    def update_layer(self, layer: Layer) -> None:
        # Weights
        layer._m = self.beta1 * layer._m + (1 - self.beta1) * layer.weights_gradient
        layer._v = self.beta2 * layer._v + (1 - self.beta2) * np.square(
            layer.weights_gradient
        )

        m_hat = layer._m / (1 - np.power(self.beta1, self.time_step))
        v_hat = layer._v / (1 - np.power(self.beta2, self.time_step))

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Bias
        layer._m_bias = (
            self.beta1 * layer._m_bias + (1 - self.beta1) * layer.bias_gradient
        )
        layer._v_bias = self.beta2 * layer._v_bias + (1 - self.beta2) * np.square(
            layer.bias_gradient
        )

        m_hat_bias = layer._m_bias / (1 - np.power(self.beta1, self.time_step))
        v_hat_bias = layer._v_bias / (1 - np.power(self.beta2, self.time_step))

        layer.bias -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + 1e-8)

    def initialize_adam_attrs(self, layer: Layer) -> None:
        layer._m = np.zeros_like(layer.weights)
        layer._v = np.zeros_like(layer.weights)
        layer._m_bias = np.zeros_like(layer.bias)
        layer._v_bias = np.zeros_like(layer.bias)

    def __str__(self) -> str:
        return (
            f"Adam(learning_rate={self.learning_rate},"
            f" beta1={self.beta1}, beta2={self.beta2})"
        )

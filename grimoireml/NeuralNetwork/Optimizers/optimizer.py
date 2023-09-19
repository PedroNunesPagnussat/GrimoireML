import numpy as np
from abc import ABC, abstractmethod
from grimoireml.NeuralNetwork.Layers.layer import Layer


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update_params(self, layers: np.array) -> None:
        """ This updates all the layers in the model. """
        for layer in layers:
            self.update_layer(layer)

    @abstractmethod
    def update_layer(self, layer: Layer) -> None:
        """ This updates the layer. """
        pass

    @abstractmethod
    def __str__(self):
        pass



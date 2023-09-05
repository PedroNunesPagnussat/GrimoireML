from abc import ABC, abstractmethod
from typing import List

from GrimoireML.grimoireml.nn.layers import Layer
from .layers import Layer 

import numpy as np

class Optimizer(ABC):
    def __init__(self, lr):
        self._lr = lr


    def _update(self, layers: List[Layer]):
        for layer in layers[1:]:
            self._layer_update(layer)


    @abstractmethod
    def _layer_update(self, layers: List[Layer]):
        pass

class SGD(Optimizer):
    def __init__(self, lr: float):
        super().__init__(lr)

    
    def _layer_update(self, layer: Layer):
        layer._weights -= self._lr * layer._weights_grad
        layer._biases -= self._lr * layer._bias_grad
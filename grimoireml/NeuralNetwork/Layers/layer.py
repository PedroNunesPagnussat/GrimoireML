from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """This is the template for all layers"""

    def __init__(self, output_shape: int, input_shape: tuple = None):
        """This is the constructor for the Layer class"""
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abstractmethod
    def __call__(self, input_layer: "Layer") -> "Layer":
        """This method will be called when the object is called"""
        pass

    @abstractmethod
    def _forward(self, input_data: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        pass

    @abstractmethod
    def _backward(self, output_data: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        pass

    @abstractmethod
    def predict(self, learning_rate: float) -> None:
        """This will be used to predict the output of the layer"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

from abc import ABC, abstractmethod
import numpy as np
from grimoireml.Functions.function import Function


class LossFunction(Function, ABC):
    """This is the template for all loss functions"""

    @abstractmethod
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        pass

    @abstractmethod
    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        pass

    # @abstractmethod
    # def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    #     """This method will be called when the object is called"""
    #     pass

    @abstractmethod
    def __str__(self) -> str:
        """This method will be considered the name of the function"""
        pass

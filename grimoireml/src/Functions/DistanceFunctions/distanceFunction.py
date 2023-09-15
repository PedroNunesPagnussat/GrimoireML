from abc import ABC, abstractmethod
import numpy as np
from src.Functions.function import Function


class DistanceFunction(Function):
    """This is the base class for all distance functions"""

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """This method is what calculates the distance between two points"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """This method is called when the function is printed"""
        pass
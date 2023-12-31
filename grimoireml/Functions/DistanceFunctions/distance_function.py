from abc import ABC, abstractmethod
import numpy as np
from grimoireml.Functions.function import Function


class DistanceFunction(Function, ABC):
    """This is the base class for all distance functions"""

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """This method is what calculates the distance between two points"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """This method is called when the function is printed"""
        pass

    def within_range(self, x: np.array, y: np.array, threshold) -> bool:
        """This method checks if the distance between two points is within a threshold"""
        if y.ndim == 1:
            self(x, y) <= threshold

        return np.where(self(x, y) <= threshold, True, False)

    def distance_matrix(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """This method calculates the distance matrix between two sets of points"""

        if y is None:
            y = x

        return np.array([[self(x_i, y_j) for y_j in y] for x_i in x])

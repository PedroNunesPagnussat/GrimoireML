from grimoireml.Functions.DistanceFunctions.distance_function import DistanceFunction
import numpy as np


class ManhattanDistance(DistanceFunction):
    """This class represents the Manhattan Distance function"""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """This method is what calculates the distance between two points"""
        if y.ndim == 1:
            return np.sum(np.abs(x - y))
        return np.sum(np.abs(x - y), axis=1)

    def __str__(self) -> str:
        """This method is called when the function is printed"""
        return "Manhattan Distance"
from src.Functions.DistanceFunctions.distance_function import DistanceFunction
import numpy as np


class EuclidianDistance(DistanceFunction):
    """This class represents the Euclidian distance function"""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """This method is what calculates the distance between two points"""
        return np.linalg.norm(x - y)

    def __str__(self) -> str:
        """This method is called when the function is printed"""
        return "Euclidian Distance"
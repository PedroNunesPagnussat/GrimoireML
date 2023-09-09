from abc import ABC, abstractmethod
import numpy as np

class DistanceFunction(ABC):
    """
    Abstract base class for distance functions.
    
    This class defines the interface that all distance functions must implement.
    The actual distance computation is done in the private method `_compute`.
    """
    
    @abstractmethod
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the distance between vectors x and y (private method).
        
        Args:
            x (np.ndarray): The first vector.
            y (np.ndarray): The second vector.
        
        Returns:
            np.ndarray: The computed distance.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the string representation of the distance function.
        
        Returns:
            str: The name or description of the distance function.
        """
        pass

class EuclideanDistance(DistanceFunction):
    """
    Class for Euclidean distance computation.
    """
    
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Euclidean distance between vectors x and y.
        
        Args:
            x (np.ndarray): The first vector.
            y (np.ndarray): The second vector.
        
        Returns:
            np.ndarray: The Euclidean distance between x and y.
        """
        return np.linalg.norm(x - y, axis=1)

    def __str__(self) -> str:
        return "Euclidean Distance"

class ManhattanDistance(DistanceFunction):
    """
    Class for Manhattan distance computation.
    """
    
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Manhattan distance between vectors x and y.
        
        Args:
            x (np.ndarray): The first vector.
            y (np.ndarray): The second vector.
        
        Returns:
            np.ndarray: The Manhattan distance between x and y.
        """
        return np.sum(np.abs(x - y), axis=1)

    def __str__(self) -> str:
        return "Manhattan Distance"

class CosineSimilarity(DistanceFunction):
    """
    Class for Cosine similarity computation.
    """
    
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Cosine similarity between vectors x and y.
        
        Args:
            x (np.ndarray): The first vector.
            y (np.ndarray): The second vector.
        
        Returns:
            np.ndarray: The Cosine similarity between x and y.
        """
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y, axis=1)
        return np.dot(y, x) / (y_norm * x_norm)

    def __str__(self) -> str:
        return "Cosine Similarity"
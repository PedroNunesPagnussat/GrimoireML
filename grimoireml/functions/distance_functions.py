from typing import Callable
import numpy as np

def get_distance_function(distance: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Return distance function based on input string.
    
    Args:
        distance (str): The name of the distance function ("euclidean", "manhattan", "cosine").
        
    Returns:
        Callable: The distance function.
        
    Raises:
        Exception: If an invalid distance function name is provided.
    """
    if distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_similarity
    raise Exception("Invalid distance function")

def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between two vectors x and y.
    
    Args:
        x (np.ndarray): The first vector.
        y (np.ndarray): The second vector.
        
    Returns:
        np.ndarray: The Euclidean distance between x and y.
    """
    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Manhattan distance between two vectors x and y.
    
    Args:
        x (np.ndarray): The first vector.
        y (np.ndarray): The second vector.
        
    Returns:
        np.ndarray: The Manhattan distance between x and y.
    """
    return np.sum(np.abs(x - y), axis=1)

def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Cosine similarity between vector x and each row in matrix y.
    
    Args:
        x (np.ndarray): The first vector.
        y (np.ndarray): The matrix containing multiple vectors.
        
    Returns:
        np.ndarray: The Cosine similarity between x and each row in y.
    """
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)
    return np.dot(y, x) / (y_norm * x_norm)

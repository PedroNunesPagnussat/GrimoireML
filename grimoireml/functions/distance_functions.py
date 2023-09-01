import numpy as np
from scipy.spatial import distance

def get_distance_function(distance):
    """Return distance function based on input string."""
    if distance == "euclidean":
        return _euclidean_distance
    elif distance == "manhattan":
        return _manhattan_distance
    elif distance == "cosine":
        return _cosine_similarity
    raise Exception("Invalid distance function")

def _euclidean_distance(x, y):
    """Calculate Euclidean distance between two vectors x and y."""
    # Using numpy's built-in function for efficiency

    return np.linalg.norm(x - y, axis=1)

def _manhattan_distance(x, y):
    """Calculate Manhattan distance between two vectors x and y."""
    # Using numpy's built-in function for efficiency
    return np.sum(np.abs(x - y), axis=1)

def _cosine_similarity(x, y):
    """Calculate Cosine similarity between vector x and each row in matrix y."""
    # Normalize x and y for cosine similarity
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)
    
    # Using numpy's dot product and broadcasting for efficiency
    return np.dot(y, x) / (y_norm * x_norm)

from typing import List, Union
import numpy as np
from ..functions.distance_functions import DistanceFunction

class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    
    Attributes:
        _min_points (int): The minimum number of points to form a dense region.
        _points (List[Point]): List of Point objects representing the data points.
        dist_func (Callable): Distance function to use.
        _eps (float): The radius of the neighborhood.
        n_clusters (int): The number of clusters formed.
        points_array (np.ndarray): Numpy array of data points.
    """
    
    class Point:
        """
        Inner class to represent a data point.
        
        Attributes:
            data_point (np.ndarray): The data point.
            cluster (int): The cluster to which the point belongs.
            visited (bool): Whether the point has been visited.
        """
        def __init__(self, data_point: np.ndarray):
            self.data_point = data_point
            self.cluster = -1
            self.visited = False

    def __init__(self, eps: float = 0.5, min_points: int = 5, dist_func: str = DistanceFunction):
        """
        Initialize DBSCAN with epsilon, minimum points, and distance function.
        
        Args:
            eps (float): The radius of the neighborhood.
            min_points (int): The minimum number of points to form a dense region.
            dist_func (str): The name of the distance function to use.
        """
        self._min_points = min_points
        self._points = None
        self.dist_func = dist_func
        self._eps = eps
        self.n_clusters = 0
        self.points_array = None

    def fit(self, data: np.ndarray) -> List[int]:
        """
        Fit the model to the data.
        
        Args:
            data (np.ndarray): The data to cluster.
        
        Returns:
            List[int]: Cluster labels for each point.
        """
        self._points = np.array([self.Point(point) for point in data])
        self.points_array = np.array([p.data_point for p in self._points]) 
        self._dbscan()
        return self.get_clusters()

    def get_clusters(self) -> List[int]:
        """
        Get the cluster labels for each point.
        
        Returns:
            List[int]: Cluster labels for each point.
        """
        return [point.cluster for point in self._points]

    def _dbscan(self):
        """Perform the DBSCAN clustering."""
        for point in self._points:
            if point.visited:
                continue
            point.visited = True
            neighbors = self._get_neighbors(point)             
            if len(neighbors) < self._min_points:
                continue

            self._expand_cluster(point, neighbors)
            self.n_clusters += 1

    def _get_neighbors(self, point: 'DBSCAN.Point') -> List['DBSCAN.Point']:
        """
        Get neighbors of a given point within epsilon distance.
        
        Args:
            point (DBSCAN.Point): The point for which to find neighbors.
        
        Returns:
            List[DBSCAN.Point]: List of neighbor points.
        """
        distances = self.dist_func._compute(self.points_array, point.data_point)
        distances[np.isclose(distances, 0)] = float('inf')
        neighbor_indices = np.where(distances <= self._eps)[0]
        return [self._points[i] for i in neighbor_indices]

    def _expand_cluster(self, point: 'DBSCAN.Point', neighbors: List['DBSCAN.Point']):
        """
        Expand the cluster from a given point.
        
        Args:
            point (DBSCAN.Point): The point from which to expand the cluster.
            neighbors (List[DBSCAN.Point]): The neighbors of the point.
        """
        unique_neighbors = set(neighbors)
        
        point.cluster = self.n_clusters
        point.visited = True

        while unique_neighbors:
            n = unique_neighbors.pop()
            if not n.visited:
                n.visited = True
                n.cluster = self.n_clusters
                new_neighbors = self._get_neighbors(n)

                if len(new_neighbors) > self._min_points:
                    unique_neighbors.update(new_neighbors)

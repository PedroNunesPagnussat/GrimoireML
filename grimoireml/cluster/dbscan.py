import numpy as np
from ..functions import distance_functions

class DBSCAN():
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)"""
    
    class Point():
        """Inner class to represent a data point."""
        def __init__(self, data_point):
            self.data_point = data_point
            self.cluster = -1
            self.visited = False
            self.neighbors = None

    def __init__(self, eps=0.5, min_points=5, dist_func="euclidean"):
        """Initialize DBSCAN with epsilon and minimum points."""
        self._eps = eps
        self._min_points = min_points
        self._points = None
        self.dist_func = distance_functions.get_distance_function(dist_func)
        self.n_clusters = 0

    def fit(self, data):
        """Fit the model to the data."""
        self._points = np.array([self.Point(point) for point in data])
        self._set_neighbors()
        self._dbscan()
        return self.get_clusters()

    def get_clusters(self):
        """Get the cluster labels for each point."""
        return [point.cluster for point in self._points]

    def _dbscan(self):
        """Perform the DBSCAN clustering."""
        for point in self._points:
            if point.visited:
                continue
            point.visited = True
            neighbors = point.neighbors
            if len(neighbors) < self._min_points:
                continue

            self._expand_cluster(point, neighbors)
            self.n_clusters += 1

    def _set_neighbors(self):
        """Pre-calculate neighbors for all points."""
        points_array = np.array([p.data_point for p in self._points])
        for point in self._points:
            point.neighbors = self._get_neighbors(point, points_array)

    def _get_neighbors(self, point, points_array):
        """Get neighbors of a given point."""
        
        distances = np.linalg.norm(points_array - point.data_point, axis=1)
        distances[np.isclose(distances, 0)] = float('inf')
        neighbor_indices = np.where(distances <= self._eps)[0]
        return [self._points[i] for i in neighbor_indices]
    
    def _expand_cluster_(self, point, neighbors):
        """Expand the cluster from a given point."""
        point.cluster = self.n_clusters
        point.visited = True
        for n in neighbors:
            if not n.visited:
                n.visited = True
                n.cluster = self.n_clusters
                new_neighbors = n.neighbors
                if len(new_neighbors) > self._min_points:
                    self._expand_cluster(n, new_neighbors)


    def _expand_cluster(self, point, neighbors):
        point.cluster = self.n_clusters
        point.visited = True

        while len(neighbors) > 0:
            n = neighbors.pop()
            if not n.visited:
                n.visited = True
                n.cluster = self.n_clusters
                new_neighbors = n.neighbors
                if len(new_neighbors) > self._min_points:
                    neighbors.extend(new_neighbors)

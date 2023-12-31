import numpy as np
from grimoireml.Functions.DistanceFunctions.distance_function import DistanceFunction
from grimoireml.Functions.DistanceFunctions import EuclideanDistance


class DBSCAN:
    """ "This class represents the:
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    clustering algorithm"""

    class Point:
        """This class represents a point in n-dimensional space"""

        def __init__(self, data: np.ndarray):
            self.data = np.array(data)
            self.cluster = -1
            self.visited = False

        def __str__(self) -> str:
            return str(self.data)

    def __init__(
        self,
        epsilon: float = 0.55,
        min_points: int = 5,
        distance_function: DistanceFunction = None,
    ):
        """This method initializes the DBScan class"""

        if distance_function is None:
            distance_function = EuclideanDistance()
        self._epsilon = epsilon
        self._min_points = min_points
        self._distance_function = distance_function
        self._points = None
        self._X = None

        self.n_clusters = None
        self.clusters = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """This method fits the data to the DBScan algorithm"""
        self._X = X
        self._points = np.array(
            [DBSCAN.Point(data) for data in self._X], dtype=DBSCAN.Point
        )
        self.n_clusters = 0
        self._dbscan()
        self.clusters = np.array([point.cluster for point in self._points])

        return self.clusters

    def _dbscan(self):
        """This method is the main DBScan algorithm"""

        for point in self._points:
            if point.visited:
                continue

            point.visited = True

            mask = self._distance_function.within_range(
                point.data, self._X, self._epsilon
            )
            neighbors = self._points[mask]
            neighbors = neighbors[neighbors != point]

            if len(neighbors) < self._min_points:
                continue

            self._expand_cluster(point, neighbors)
            self.n_clusters += 1

    def _expand_cluster(self, point: Point, neighbors: np.ndarray):
        """This spread the cluster to the neighbors of the point,
        if the points is a central point and the neighbors are not already in a cluster"""

        points_to_check = set(neighbors)

        point.cluster = self.n_clusters
        point.visited = True

        while points_to_check:
            neighbor = points_to_check.pop()

            if neighbor.visited:
                continue

            neighbor.visited = True
            neighbor.cluster = self.n_clusters

            neighbor_neighbors_mask = self._distance_function.within_range(
                neighbor.data, self._X, self._epsilon
            )
            neighbor_neighbors = self._points[neighbor_neighbors_mask]
            neighbor_neighbors = neighbor_neighbors[neighbor_neighbors != neighbor]

            if len(neighbor_neighbors) >= self._min_points:
                points_to_check.update(neighbor_neighbors)

    def __str__(self) -> str:
        """This method is called when the function is printed."""
        return (
            f"DBScan(epsilon={self._epsilon},"
            f"min_points={self._min_points}, "
            f"distance_function={self._distance_function})"
        )

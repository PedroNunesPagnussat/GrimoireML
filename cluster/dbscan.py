import numpy as np


class DBSCAN():
    def __init__(self, eps, min_points):
        self._eps = eps
        self._min_points = min_points

    def fit(self, data):        
        self._dbscan()

        return 0

    def get_clusters(self):
        pass

    def _dbscan(self):        
        pass
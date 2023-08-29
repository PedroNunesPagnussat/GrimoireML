import numpy as np

class DBSCAN():
    class Point():
        def __init__(self, data):
            self.data = data,
            self.type = None
            self.cluster = None

        def getData(self):
            return self.data[0]

    
    def __init__(self, epsilon, min_points):
        self.epsilon = epsilon
        self.min_points = min_points
        self.points = None
        # self.clusters = None
        # self.noise = None

    def fit(self, data):
        for point in data:
            a = self.Point(point)
            print(a.getData())
            exit()
        self.points = [self.Point(point) for point in data]
        for point in self.points:
            print(point.data)
            exit()
        self.clusters = self.__dbscan()

    def __dbscan(self):
        for i, point in enumerate(self.points):
            point.n_neighbors = self.__get_neighbors(point)

            if point.n_neighbors >= self.min_points:
                point.type = 1
            
            elif point.n_neighbors > 0:
                point.type = 0
            
            else:
                point.type = -1

        for point in self.points:
            print(point.type)

    def __get_neighbors(self, point):
        neighbors = 0
        for p in self.points:
            print(point.data, p.data)
            exit()
            if point != p and self.__distance(point, p) < self.eps:
                neighbors += 1
        return len(neighbors)

    def __distance(self, p1, p2):
        print(p1.data, p2.data)
        exit()
        return np.linalg.norm(p1.data - p2.data)
            
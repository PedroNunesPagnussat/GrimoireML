from grimoireml.Functions.DistanceFunctions import EuclideanDistance, ManhattanDistance
from grimoireml.Functions.EvaluationFunctions import Accuracy

from scipy.spatial import distance_matrix
import numpy as np




distance = EuclideanDistance()
arr = np.array([[1, 2], [3, 2], [3, 20]])
print(distance.distance_matrix(arr))
exit()

x = np.array([[1, 2], [3, 2], [3, 20]])
y = np.array([[1, 2], [3, 2], [3, 20]])

print(distance(x, y))
print(distance.within_range(x, y, 16))
print(distance.distance_matrix(x))

print(dist.cdist(x, y, 'cityblock'))
from grimoireml.Functions.DistanceFunctions import EuclideanDistance, ManhattanDistance
from grimoireml.Functions.EvaluationFunctions import Accuracy
from grimoireml.Cluster import DBSCAN
from grimoireml.NeuralNetwork.LossFunctions import MSELoss

from dev_data_fetch import fetch_data
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
import numpy as np
import torch
from icecream import ic




arr1 = np.array([1, 2])
arr2 = np.array([3, 2])
loss = MSELoss()
ic(loss.derivative(arr1, arr2))
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# arr2 = np.array([[1, 2, 3], [4, 5, 6]])

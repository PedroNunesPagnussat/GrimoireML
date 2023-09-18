from grimoireml.Functions.DistanceFunctions import EuclideanDistance, ManhattanDistance
from grimoireml.Functions.EvaluationFunctions import Accuracy
from grimoireml.Cluster import DBSCAN
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Layers import Dense
from dev_data_fetch import fetch_data
import numpy as np
import torch
from icecream import ic


layer = Dense(output_shape=1, input_shape=(3,))
layer.weights = np.array([[0.7], [-0.1], [0.2]])
layer._input_data = np.array([[0.567, 0.572, 0.443]])
# accumulated_error = np.ndarray([[-0.022702]])
accumulated_error = np.array([[-0.022702]])
propagate_error = layer._backward(accumulated_error)
expected_propagate_error = np.array([[-0.01588914, 0.0022702, -0.0045404]])

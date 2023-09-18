from grimoireml.Functions.DistanceFunctions import EuclideanDistance, ManhattanDistance
from grimoireml.Functions.EvaluationFunctions import Accuracy
from grimoireml.Cluster import DBSCAN
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Layers import Dense
from dev_data_fetch import fetch_data
import numpy as np
import torch
from icecream import ic


x = Dense(output_shape=5, input_shape=(3,))
ic(str(x))
x = Dense(10)(x)
ic(str(x))
x = Dense(5)(x)
ic(str(x))

from grimoireml.Functions.DistanceFunctions import EuclideanDistance, ManhattanDistance
from grimoireml.Functions.EvaluationFunctions import Accuracy
from grimoireml.Cluster import DBSCAN
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Layers import Dense, ReLU, Sigmoid
from grimoireml.NeuralNetwork.Optimizers import SGD, Adam, Adagrad
from dev_data_fetch import fetch_data
import numpy as np
from icecream import ic
from grimoireml.NeuralNetwork.Models import Sequential
from grimoireml.NeuralNetwork.Initializers.BiasInitializers import (
    ZerosBias,
    OnesBias,
    UniformBias,
    NormalBias,
    RandomBias,
    ConstantBias,
)
from grimoireml.NeuralNetwork.Initializers.WeightInitializers import (
    ZerosWeight,
    NormalWeight,
    RandomWeight,
    UniformWeight,
    HeNormalWeight,
    HeUniformWeight,
    XavierNormalWeight,
    XavierUniformWeight,
)

np.random.seed(42)

X = np.array([[0.5, 0.1]])
y = np.array([[0.7]])

model = Sequential(
    [
        Dense(3, input_shape=(2,)),
        ReLU(),
        Dense(1, weight_initializer=XavierUniformWeight()),
        Sigmoid(),
    ]
)
print(model)
model.build(MSELoss(), Adam(learning_rate=0.01))
model.fit(
    X, y, epochs=150, batch_size=1, metrics=[], validation_data=None, verbose=True
)

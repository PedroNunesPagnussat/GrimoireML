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
from grimoireml.NeuralNetwork.Initializers.BiasInitializers import ZerosBias

i = ZerosBias()
ic(i(3))


exit()
X = np.array([[0.5, 0.1]])
y = np.array([[0.7]])

model = Sequential()

x = model.add(Dense(3, input_shape=(2,)))
x.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
x.bias = np.array([[0.0, 0.0, 0.0]])
x = model.add(Sigmoid()(x))
x = model.add(Dense(1)(x))
x.weights = np.array([[0.7], [-0.1], [0.2]])
x.bias = np.array([[0.0]])
x = model.add(Sigmoid()(x))

model.build(MSELoss(), Adam(learning_rate=0.01))
model.fit(X, y, epochs=150, batch_size=1, metrics=[], validation_data=None, verbose=True)

layers = model.layers
ic(layers[2].weights)

print(model)
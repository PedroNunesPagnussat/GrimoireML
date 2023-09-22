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
from dev_data_fetch import fetch_data

X, y = fetch_data("breast_cancer")
# standart scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


input_shape = X.shape[1]



model = Sequential([
    Dense(3, input_shape=(input_shape,)),
    ReLU(),
    Dense(1),
    Sigmoid()
])
model.build(MSELoss(), Adam(learning_rate=0.01))
#Accuracy(classification_type="binary")
model.fit(X, y, epochs=2, batch_size=16, metrics=[], validation_data=None, verbose=True)
exit()

model = Sequential()

x = model.add(Dense(3, input_shape=(2,)))
x.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
x.bias = np.array([[0.0, 0.0, 0.0]])
x = model.add(Sigmoid()(x))
x = model.add(Dense(1)(x))
x.weights = np.array([[0.7], [-0.1], [0.2]])
x.bias = np.array([[0.0]])
x = model.add(Sigmoid()(x))


layers = model.layers
ic(layers[2].weights)

exit()
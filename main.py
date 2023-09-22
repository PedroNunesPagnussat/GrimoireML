from grimoireml.NeuralNetwork.Layers import ReLU, Sigmoid, Dense
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Optimizers import SGD
from grimoireml.NeuralNetwork.Models import Sequential
from dev_data_fetch import fetch_data
import numpy as np

# Fetching data
X, y = fetch_data("breast_cancer")

model = Sequential(
    [
        Dense(16, input_shape=(X.shape[1],)),
        ReLU(),
        Dense(8),
        ReLU(),
        Dense(1),
        Sigmoid(),
    ]
)

model.build(loss=MSELoss(), optimizer=SGD())

model.fit(X, y, epochs=100, batch_size=32)

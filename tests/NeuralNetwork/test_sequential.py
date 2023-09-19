import pytest
import numpy as np
from grimoireml.NeuralNetwork.Models import Sequential
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Layers import Dense, Sigmoid
from grimoireml.NeuralNetwork.Optimizers import SGD

np.random.seed(42)

@pytest.mark.skip(reason="Not implemented")
def test_simple_nn():
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

    model.build(MSELoss(), SGD(learning_rate=0.1))
    model.fit(X, y, 1, 1, [], None)

    assert np.allclose(model.history.loss_history, [0.00446782], atol=1e-3)

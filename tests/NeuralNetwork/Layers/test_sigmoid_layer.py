import pytest
import numpy as np
from grimoireml.NeuralNetwork.Layers import Dense, Sigmoid

np.random.seed(42)


def test_sigmoid_initialization():
    layer = Dense(output_shape=5, input_shape=(3,))
    test_layer = Sigmoid()(layer)
    assert test_layer.input_shape == 5
    assert test_layer.output_shape == 5


def test_sigmoid_forward_pass():
    layer = Dense(output_shape=3, input_shape=(2,))
    layer.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
    layer.bias = np.array([[0, 0, 0]])
    layer2 = Sigmoid()(layer)

    input_data = np.array([0.5, 0.1])

    out1 = layer._forward(input_data)
    output = layer2._forward(out1)

    expected_output = np.array([0.56711, 0.57199, 0.44276])

    assert np.allclose(output, expected_output, atol=1e-3)


def test_sigmoid_backward_pass():
    error = np.array([[-0.095]])
    layer = Dense(output_shape=1, input_shape=(3,))
    layer2 = Sigmoid()(layer)

    layer.weights = np.array([[0.7], [-0.1], [0.2]])

    input_data = np.array([[0.56711, 0.57199, 0.44276]])
    input_data = layer._forward(input_data)
    layer2._forward(input_data)

    error = layer2._backward(error)
    assert np.allclose(error, -0.022702, atol=1e-5)
    error = layer._backward(error)
    temp = np.multiply(error, [[0.245499, 0.244816, 0.246722]])
    assert np.allclose(temp, [[-0.0039013, 0.0005558, -0.0011202]], atol=1e-3)

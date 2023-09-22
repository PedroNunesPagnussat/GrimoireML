import pytest
import numpy as np
from grimoireml.NeuralNetwork.Layers import (
    Dense,
)  # Replace `your_module_path` with the actual module path

np.random.seed(42)


def test_dense_initialization():
    layer = Dense(output_shape=5, input_shape=(3,))
    assert layer.weights.shape == (3, 5)
    assert layer.bias.shape == (1, 5)


def test_dense_initialization_no_input_shape():
    layer = Dense(output_shape=5)
    x = Dense(output_shape=5, input_shape=(3,))
    x = Dense(10)(x)

    assert layer.weights is None
    assert layer.bias is None

    assert x.weights.shape == (5, 10)
    assert x.bias.shape == (1, 10)


def test_dense_forward_pass():
    layer = Dense(output_shape=3, input_shape=(2,))
    layer.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
    layer.bias = np.array([[0, 0, 0]])
    input_data = np.array([0.5, 0.1])

    output = layer.forward(input_data)
    expected_output = np.array([0.27, 0.29, -0.23])

    assert np.allclose(output, expected_output, atol=1e-5)


def test_dense_backward_pass():
    layer = Dense(output_shape=1, input_shape=(3,))
    layer.weights = np.array([[0.7], [-0.1], [0.2]])
    layer.input_data = np.array([[0.567, 0.572, 0.443]])
    accumulated_error = np.array([[-0.022702]])
    propagate_error = layer.backward(accumulated_error)
    expected_propagate_error = np.array([[-0.01588914, 0.0022702, -0.0045404]])

    assert np.allclose(propagate_error, expected_propagate_error, atol=1e-5)
    assert np.allclose(
        layer.weights_gradient,
        np.array([[-0.01288914], [0.0022702], [-0.0045404]]),
        atol=0.016,
    )

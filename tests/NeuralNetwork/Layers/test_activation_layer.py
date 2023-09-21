import pytest
import numpy as np
from grimoireml.NeuralNetwork.Layers import (
    Dense,
    Sigmoid,
    Linear,
    Tanh,
    LeakyReLU,
    ReLU,
)

np.random.seed(42)


def test_activation_initialization():
    layer = Dense(output_shape=5, input_shape=(3,))
    test_layer = Sigmoid()(layer)
    assert test_layer.input_shape == 5
    assert test_layer.output_shape == 5


def test_activation_forward_pass():
    layer = Dense(output_shape=3, input_shape=(2,))
    layer.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
    layer.bias = np.array([[0, 0, 0]])
    layer2 = Sigmoid()(layer)

    input_data = np.array([0.5, 0.1])

    out1 = layer.forward(input_data)
    output = layer2.forward(out1)

    expected_output = np.array([0.56711, 0.57199, 0.44276])

    assert np.allclose(output, expected_output, atol=1e-3)


def test_activation_backward_pass():
    error = np.array([[-0.095]])
    layer = Dense(output_shape=1, input_shape=(3,))
    layer2 = Sigmoid()(layer)

    layer.weights = np.array([[0.7], [-0.1], [0.2]])

    input_data = np.array([[0.56711, 0.57199, 0.44276]])
    input_data = layer.forward(input_data)
    layer2.forward(input_data)

    error = layer2.backward(error)
    assert np.allclose(error, -0.022702, atol=1e-5)
    error = layer.backward(error)
    temp = np.multiply(error, [[0.245499, 0.244816, 0.246722]])
    assert np.allclose(temp, [[-0.0039013, 0.0005558, -0.0011202]], atol=1e-3)


def test_sigmoid():
    activation = Sigmoid().activation
    assert np.allclose(
        activation(np.array([-1])), np.array([0.26894142136992605]), atol=1e-8
    )
    assert np.allclose(
        activation(np.array([-0.5])), np.array([0.37754066879810416]), atol=1e-8
    )
    assert np.allclose(activation(np.array([0])), np.array([0.5]), atol=1e-8)
    assert np.allclose(
        activation(np.array([0.5])), np.array([0.6224593312018951]), atol=1e-8
    )
    assert np.allclose(
        activation(np.array([1])), np.array([0.7310585786300049]), atol=1e-8
    )


def test_sigmoid_derivative():
    activation_derivative = Sigmoid().derivative
    assert np.allclose(
        activation_derivative(np.array([-1])),
        np.array([0.19661193324148185]),
        atol=1e-8,
    )
    assert np.allclose(
        activation_derivative(np.array([-0.5])),
        np.array([0.2350037122015945]),
        atol=1e-8,
    )
    assert np.allclose(
        activation_derivative(np.array([0])), np.array([0.25]), atol=1e-8
    )
    assert np.allclose(
        activation_derivative(np.array([0.5])),
        np.array([0.2350037122015945]),
        atol=1e-8,
    )
    assert np.allclose(
        activation_derivative(np.array([1])), np.array([0.19661193324148185]), atol=1e-8
    )


def test_tanh():
    activation = Tanh().activation
    assert np.allclose(
        activation(np.array([-1])), np.array([-0.7615941559557649]), atol=1e-8
    )
    assert np.allclose(
        activation(np.array([-0.5])), np.array([-0.46211715726000974]), atol=1e-8
    )
    assert np.allclose(activation(np.array([0])), np.array([0]), atol=1e-8)
    assert np.allclose(
        activation(np.array([0.5])), np.array([0.46211715726000974]), atol=1e-8
    )
    assert np.allclose(
        activation(np.array([1])), np.array([0.7615941559557649]), atol=1e-8
    )


def test_tanh_derivative():
    activation_derivative = Tanh().derivative
    assert np.allclose(
        activation_derivative(np.array([-1])),
        np.array([0.41997434161402614]),
        atol=1e-8,
    )
    assert np.allclose(
        activation_derivative(np.array([-0.5])),
        np.array([0.7864477329659274]),
        atol=1e-8,
    )
    assert np.allclose(activation_derivative(np.array([0])), np.array([1]), atol=1e-8)
    assert np.allclose(
        activation_derivative(np.array([0.5])),
        np.array([0.7864477329659274]),
        atol=1e-8,
    )
    assert np.allclose(
        activation_derivative(np.array([1])), np.array([0.41997434161402614]), atol=1e-8
    )


def test_relu():
    activation = ReLU().activation
    assert np.allclose(activation(np.array([-1])), np.array([0]), atol=1e-8)
    assert np.allclose(activation(np.array([-0.5])), np.array([0]), atol=1e-8)
    assert np.allclose(activation(np.array([0])), np.array([0]), atol=1e-8)
    assert np.allclose(activation(np.array([0.5])), np.array([0.5]), atol=1e-8)
    assert np.allclose(activation(np.array([1])), np.array([1]), atol=1e-8)


def test_relu_derivative():
    activation_derivative = ReLU().derivative
    assert np.allclose(activation_derivative(np.array([-1])), np.array([0]), atol=1e-8)
    assert np.allclose(
        activation_derivative(np.array([-0.5])), np.array([0]), atol=1e-8
    )
    assert np.allclose(activation_derivative(np.array([0])), np.array([0]), atol=1e-8)
    assert np.allclose(activation_derivative(np.array([0.5])), np.array([1]), atol=1e-8)
    assert np.allclose(activation_derivative(np.array([1])), np.array([1]), atol=1e-8)


def test_leaky_relu():
    activation = LeakyReLU().activation

    assert np.allclose(activation(np.array([-1])), np.array([-0.01]), atol=1e-8)
    assert np.allclose(activation(np.array([-0.5])), np.array([-0.005]), atol=1e-8)
    assert np.allclose(activation(np.array([0])), np.array([0]), atol=1e-8)
    assert np.allclose(activation(np.array([0.5])), np.array([0.5]), atol=1e-8)
    assert np.allclose(activation(np.array([1])), np.array([1]), atol=1e-8)

    activation = LeakyReLU(alpha=0.1).activation
    assert np.allclose(activation(np.array([-1])), np.array([-0.1]), atol=1e-8)
    assert np.allclose(activation(np.array([-0.5])), np.array([-0.05]), atol=1e-8)
    assert np.allclose(activation(np.array([0])), np.array([0]), atol=1e-8)
    assert np.allclose(activation(np.array([0.5])), np.array([0.5]), atol=1e-8)
    assert np.allclose(activation(np.array([1])), np.array([1]), atol=1e-8)


def test_leaky_relu_derivative():
    activation_derivative = LeakyReLU().derivative

    assert np.allclose(
        activation_derivative(np.array([-1])), np.array([0.01]), atol=1e-8
    )
    assert np.allclose(
        activation_derivative(np.array([-0.5])), np.array([0.01]), atol=1e-8
    )
    assert np.allclose(
        activation_derivative(np.array([0])), np.array([0.01]), atol=1e-8
    )
    assert np.allclose(activation_derivative(np.array([0.5])), np.array([1]), atol=1e-8)
    assert np.allclose(activation_derivative(np.array([1])), np.array([1]), atol=1e-8)

    activation_derivative = LeakyReLU(alpha=0.1).derivative
    assert np.allclose(
        activation_derivative(np.array([-1])), np.array([0.1]), atol=1e-8
    )
    assert np.allclose(
        activation_derivative(np.array([-0.5])), np.array([0.1]), atol=1e-8
    )
    assert np.allclose(activation_derivative(np.array([0])), np.array([0.1]), atol=1e-8)
    assert np.allclose(activation_derivative(np.array([0.5])), np.array([1]), atol=1e-8)
    assert np.allclose(activation_derivative(np.array([1])), np.array([1]), atol=1e-8)


def test_linear():
    activation = Linear().activation
    assert activation(np.array([0])) == 0
    assert activation(np.array([1])) == 1
    assert activation(np.array([-1])) == -1
    assert activation(np.array([0.5])) == 0.5


def test_linear_derivative():
    activation_derivative = Linear().derivative
    assert activation_derivative(np.array([0])) == 1
    assert activation_derivative(np.array([-1])) == 1
    assert activation_derivative(np.array([0.5])) == 1
    assert activation_derivative(np.array([1])) == 1

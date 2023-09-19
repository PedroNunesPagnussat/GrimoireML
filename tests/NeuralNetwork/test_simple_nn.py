import pytest
import numpy as np
from grimoireml.NeuralNetwork.LossFunctions import MSELoss
from grimoireml.NeuralNetwork.Layers import Dense, Sigmoid

np.random.seed(42)


def test_simple_nn():

    layer1 = Dense(output_shape=3, input_shape=(2,))
    layer2 = Sigmoid()(layer1)
    layer3 = Dense(output_shape=1)(layer2)
    layer4 = Sigmoid()(layer3)
    loss = MSELoss()

    X = np.array([[0.5, 0.1]])
    y = np.array([[0.7]])

    layer1.weights = np.array([[0.5, 0.6, -0.4], [0.2, -0.1, -0.3]])
    layer1.bias = np.array([[0, 0, 0]])
    layer3.weights = np.array([[0.7], [-0.1], [0.2]])
    layer3.bias = np.array([[0]])

    out1 = layer1._forward(X)
    assert np.allclose(out1, [[0.27, 0.29, -0.23]], atol=1e-4)
    out2 = layer2._forward(out1)
    assert np.allclose(out2, [[0.567, 0.573, 0.443]], atol=1e-3)
    out3 = layer3._forward(out2)
    assert np.allclose(out3, [[0.4283]], atol=1e-4)
    final_output = layer4._forward(out3)
    assert np.allclose(final_output, [[0.605]], atol=1e-3)

    # I half the value to match professor Lucas Half MSE Loss Example
    # This can be changed once HalfMSELoss is implemented
    loss_value = loss(y, final_output) / 2

    assert np.allclose(loss_value, 0.00446782, atol=1e-3)

    # Start Backpropagation

    error = loss.derivative(final_output, y) / 2
    assert np.allclose(error, [[-0.095]], atol=1e-3)

    outB4 = layer4._backward(error)
    assert np.allclose(outB4, [[-0.022702]], atol=1e-3)
    outB3 = layer3._backward(outB4)
    assert np.allclose(outB3, [[-0.01580641, 0.00225806, -0.00451612]], atol=1e-3)
    outB2 = layer2._backward(outB3)
    assert np.allclose(outB2, [[-0.0039013, 0.0005558, -0.0011202]], atol=1e-3)
    _ = layer1._backward(outB2)

    # Compare weight gradients
    l1_w_gradient = layer1.weights_gradient
    assert np.allclose(
        l1_w_gradient,
        [[-0.00195, 0.00027, -0.00027], [-0.00039, 0.00005, -0.00011]],
        atol=1e-3,
    )

    l3_w_gradient = layer3.weights_gradient
    assert np.allclose(l3_w_gradient, [[-0.01287], [-0.012985], [-0.010051]], atol=1e-3)

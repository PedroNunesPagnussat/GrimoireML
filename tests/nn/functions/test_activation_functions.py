import pytest
from grimoireml.nn.functions.activation_functions import Sigmoid, ReLU, Tanh, Softmax, Linear, LeakyReLU
import numpy as np

def test_sigmoid():
    sigmoid = Sigmoid()
    x = np.array([0, 1, -1])
    assert np.allclose(sigmoid._compute(x), np.array([0.5, 0.73105858, 0.26894142]))
    assert np.allclose(sigmoid._compute_derivative(x), np.array([0.25, 0.19661193, 0.19661193]))

def test_relu():
    relu = ReLU()
    x = np.array([0, 1, -1])
    assert np.allclose(relu._compute(x), np.array([0, 1, 0]))
    assert np.allclose(relu._compute_derivative(x), np.array([0, 1, 0]))

def test_tanh():
    tanh = Tanh()
    x = np.array([0, 1, -1])
    assert np.allclose(tanh._compute(x), np.array([0, 0.76159416, -0.76159416]))
    assert np.allclose(tanh._compute_derivative(x), np.array([1, 0.41997434, 0.41997434]))

def test_softmax():
    softmax = Softmax()
    x = np.array([[1, 2], [1, 2]])
    assert np.allclose(softmax._compute(x), np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))
    assert np.allclose(softmax._compute_derivative(x), np.array([[0.19661193, 0.19661193], [0.19661193, 0.19661193]]))

def test_linear():
    linear = Linear()
    x = np.array([0, 1, -1])
    assert np.allclose(linear._compute(x), np.array([0, 1, -1]))
    assert np.allclose(linear._compute_derivative(x), np.array([1, 1, 1]))

def test_leaky_relu():
    leaky_relu = LeakyReLU(alpha=0.01)
    x = np.array([0, 1, -1])
    assert np.allclose(leaky_relu._compute(x), np.array([0, 1, -0.01]))
    assert np.allclose(leaky_relu._compute_derivative(x), np.array([0.01, 1, 0.01]))

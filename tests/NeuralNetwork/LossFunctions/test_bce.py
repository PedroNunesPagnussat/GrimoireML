import numpy as np
import pytest
import sys
import os
from grimoireml.NeuralNetwork.LossFunctions import BCELoss  # Import the BCELoss class

# Add the project root directory to the sys.path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(
    os.path.join(current_script_directory, "..", "..", "..")
)
sys.path.append(project_root_directory)

# Test the BCE loss function
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([[0.5]]), np.array([[1]]), 0.6931),
        (np.array([[0.9]]), np.array([[1]]), 0.1054),
        (np.array([[0.1]]), np.array([[0]]), 0.1054),
        # Add more test cases here
    ],
)
def test_bce_loss(x, y, expected):
    loss = BCELoss()
    assert loss(x, y) == pytest.approx(expected, 0.01)

# Test the derivative of the BCE loss function
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([[0.5]]), np.array([[1]]), np.array([[-2.0]])),
        (np.array([[0.9]]), np.array([[1]]), np.array([[-1.1111]])),
        (np.array([[0.1]]), np.array([[0]]), np.array([[1.1111]])),
        # Add more test cases here
    ],
)
def test_bce_loss_derivative(x, y, expected):
    loss = BCELoss()
    assert loss.derivative(x, y) == pytest.approx(expected, 0.01)

# Test the string representation of the BCE loss function
def test_str():
    assert str(BCELoss()) == "Binary Cross Entropy"

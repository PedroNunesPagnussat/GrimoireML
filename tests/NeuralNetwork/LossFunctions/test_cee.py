import numpy as np
import pytest
import sys
import os
from grimoireml.NeuralNetwork.LossFunctions import CCELoss  # Import the CCELoss class

# Add the project root directory to the sys.path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(
    os.path.join(current_script_directory, "..", "..", "..")
)
sys.path.append(project_root_directory)

# Test the CCE loss function


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([[0.2, 0.4, 0.4]]), np.array([[0, 1, 0]]), 0.9163),
        (np.array([[0.7, 0.2, 0.1]]), np.array([[1, 0, 0]]), 0.3567),
        # Add more test cases here
    ],
)
def test_cce_loss(x, y, expected):
    loss = CCELoss()
    assert loss(x, y) == pytest.approx(expected, 0.01)


# Test the derivative of the CCE loss function
@pytest.mark.parametrize(
    "x, y, expected",
    [
        (
            np.array([[0.2, 0.4, 0.4]]),
            np.array([[0, 1, 0]]),
            np.array([[0.2, -0.6, 0.4]]),
        ),
        (
            np.array([[0.7, 0.2, 0.1]]),
            np.array([[1, 0, 0]]),
            np.array([[-0.3, 0.2, 0.1]]),
        ),
        # Add more test cases here
    ],
)
def test_cce_loss_derivative(x, y, expected):
    loss = CCELoss()
    assert loss.derivative(x, y) == pytest.approx(expected, 0.01)


# Test the string representation of the CCE loss function
def test_str():
    assert str(CCELoss()) == "Categorical Cross Entropy"

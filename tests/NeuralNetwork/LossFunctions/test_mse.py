import numpy as np
import pytest
import sys
import os

current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(os.path.join(current_script_directory, '..', '..', '..'))
sys.path.append(project_root_directory)

from grimoireml.NeuralNetwork.LossFunctions import MSELoss


@pytest.mark.parametrize("x, y, expected", [
    (np.array([1]), np.array([1]), 0),

    (np.array([1, 2]), np.array([1, 2]), 0),
    (np.array([1, 2]), np.array([3, 2]), 2),
    (np.array([1, 2]), np.array([2, 3]), 1),
    (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0),

    (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0),
    (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 31.5),

    (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), 0),
    (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[3, 2, 1], [7, 8, 9]]), 5.833),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[3, 2, 1], [7, 8, 9], [7, 8, 9]]), 3.888666)

])

def test_mae_loss(x, y, expected):
    from icecream import ic
    loss = MSELoss()
    assert loss(x, y) == pytest.approx(expected, 0.01)


# @pytest.mark.parametrize("x, y, expected", [


# def test_mae_loss_derivative():


def test_str():
    assert str(MSELoss()) == "Mean Squared Error"
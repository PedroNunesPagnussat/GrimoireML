import numpy as np
import pytest
import sys
import os

#print the path that python looks for modules in
# Get the current script directory


current_script_directory = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (project_root) and add it to sys.path
project_root_directory = os.path.abspath(os.path.join(current_script_directory, '..', '..', '..'))
sys.path.append(project_root_directory)
from grimoireml.Functions.DistanceFunctions import ManhattanDistance  # Assuming this is the correct import path

@pytest.mark.parametrize("x, y, expected", [
    (np.array([1]), np.array([1]), 0),
    (np.array([1, 2]), np.array([1, 2]), 0),
    (np.array([1, 2]), np.array([3, 2]), 2),
    (np.array([2, 3]), np.array([3, 2]), 2),
    (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0),
    (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 22)
])


def test_Manhattan_distancem_single_point(x, y, expected):
    Manhattan_distancem = ManhattanDistance()
    assert Manhattan_distancem(x, y) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize("x, y, expected", [
    (np.array([1, 2]), np.array([[1, 2], [1, 2]]), np.array([0, 0])),
    (np.array([1, 2]), np.array([[2, 4], [4, 8]]), np.array([3, 9])),
])

def test_Manhattan_distancem_multiple_points(x, y, expected):
    Manhattan_distancem = ManhattanDistance()
    assert np.all(Manhattan_distancem(x, y) == pytest.approx(expected, 0.01))


@pytest.mark.parametrize("x, y, threshold, expected", [
    (np.array([1, 2]), np.array([1, 2]), 0, True),
    (np.array([1, 2]), np.array([3, 2]), 2, True),
    (np.array([2, 3]), np.array([3, 2]), 2, True),
    (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0, True),
    (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 22, True),
    (np.array([1, 2]), np.array([3, 2]), 1, False),
    (np.array([2, 3]), np.array([3, 2]), 1, False),
    (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 21, False),
])

def test_Manhattan_distancem_within_range_single_point(x, y, threshold, expected):
    Manhattan_distancem = ManhattanDistance()
    assert Manhattan_distancem.within_range(x, y, threshold) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize("x, y, threshold, expected", [
    (np.array([1, 2]), np.array([[1, 2], [1, 2]]), 0, np.array([True, True])),
    (np.array([1, 2]), np.array([[2, 4], [4, 8]]), 3, np.array([True, False])),
    (np.array([1, 2]), np.array([[2, 4], [4, 8]]), 10, np.array([True, True])),
])

def test_Manhattan_distancem_within_range_multiple_points(x, y, threshold, expected):
    from icecream import ic
    ic(x, y, threshold, expected)
    Manhattan_distancem = ManhattanDistance()
    assert np.all(Manhattan_distancem.within_range(x, y, threshold) == pytest.approx(expected, 0.01))


@pytest.mark.parametrize("x, y, expected", [
    (np.array([[1, 2], [3, 2], [3, 20]]), None, np.array([
        [0, 2, 20],
        [2, 0, 18],
        [20, 18, 0]
    ])),

    (np.array([[1, 2], [3, 2], [3, 20]]), np.array([[10, 2], [3, 2], [3, 20]]), np.array([
        [9, 2, 20],
        [7, 0, 18],
        [25, 18, 0]
    ])),
])



def test_Manhattan_distancem_matrix(x, y, expected):
    Manhattan_distancem = ManhattanDistance()
    assert np.all(Manhattan_distancem.distance_matrix(x, y) == pytest.approx(expected, 0.01))


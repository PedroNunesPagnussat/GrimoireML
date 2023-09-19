import numpy as np
import pytest
import sys
import os


current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(
    os.path.join(current_script_directory, "..", "..", "..")
)
sys.path.append(project_root_directory)


from grimoireml.Functions.DistanceFunctions import (
    EuclideanDistance,
)  # noeq: E402


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1]), np.array([1]), 0),
        (np.array([1, 2]), np.array([1, 2]), 0),
        (np.array([1, 2]), np.array([3, 2]), 2),
        (np.array([2, 3]), np.array([3, 2]), 1.4142),
        (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0),
        (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 11.2249),
    ],
)
def test_euclidean_distance_single_point(x, y, expected):
    euclidean_distance = EuclideanDistance()
    assert euclidean_distance(x, y) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 2]), np.array([[1, 2], [1, 2]]), np.array([0, 0])),
        (np.array([1, 2]), np.array([[2, 4], [4, 8]]), np.array([2.236, 6.7082])),
    ],
)
def test_euclidean_distance_multiple_points(x, y, expected):
    euclidean_distance = EuclideanDistance()
    assert np.all(euclidean_distance(x, y) == pytest.approx(expected, 0.01))


@pytest.mark.parametrize(
    "x, y, threshold, expected",
    [
        (np.array([1, 2]), np.array([1, 2]), 0, True),
        (np.array([1, 2]), np.array([3, 2]), 2, True),
        (np.array([2, 3]), np.array([3, 2]), np.sqrt(2), True),
        (np.array([4, 5, 6, 7]), np.array([4, 5, 6, 7]), 0, True),
        (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), np.sqrt(126), True),
        (np.array([1, 2]), np.array([3, 2]), 1, False),
        (np.array([2, 3]), np.array([3, 2]), 1.4141, False),
        (np.array([4, 5, 6, 7]), np.array([8, 10, 12, 14]), 11.2248, False),
    ],
)
def test_euclidean_distance_within_range_single_point(x, y, threshold, expected):
    euclidean_distance = EuclideanDistance()
    assert euclidean_distance.within_range(x, y, threshold) == pytest.approx(
        expected, 0.01
    )


@pytest.mark.parametrize(
    "x, y, threshold, expected",
    [
        (np.array([1, 2]), np.array([[1, 2], [1, 2]]), 0, np.array([True, True])),
        (np.array([1, 2]), np.array([[2, 4], [4, 8]]), 6.71, np.array([True, True])),
        (
            np.array([1, 2]),
            np.array([[2, 3], [3, 2]]),
            np.sqrt(2) + 0.1,
            np.array([True, False]),
        ),
        (
            np.array([4, 5, 6, 7]),
            np.array([[4, 5, 6, 7], [8, 10, 12, 14]]),
            np.sqrt(126) - 1,
            np.array([True, False]),
        ),
    ],
)
def test_euclidean_distance_within_range_multiple_points(x, y, threshold, expected):
    euclidean_distance = EuclideanDistance()
    # fmt: off
    assert np.all(
        euclidean_distance.within_range(x, y, threshold) == pytest.approx(expected, 0.01)
    )
    # fmt: on


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (
            np.array([[1, 2], [3, 2], [3, 20]]),
            None,
            np.array([[0.0, 2.0, 18.11077], [2.0, 0, 18.0], [18.11077, 18.0, 0.0]]),
        ),
        (
            np.array([[1, 2], [3, 2], [3, 20]]),
            np.array([[10, 2], [3, 2], [3, 20]]),
            np.array([[9.0, 2.0, 18.11077], [7.0, 0, 18.0], [19.31320792, 18.0, 0.0]]),
        ),
    ],
)
def test_euclidean_distance_matrix(x, y, expected):
    euclidean_distance = EuclideanDistance()
    assert np.all(
        euclidean_distance.distance_matrix(x, y) == pytest.approx(expected, 0.01)
    )


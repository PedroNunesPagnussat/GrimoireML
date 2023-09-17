import pytest
import numpy as np

import os
import sys
from grimoireml.Functions.EvaluationFunctions import Accuracy


current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(
    os.path.join(current_script_directory, "..", "..", "..", "..")
)
sys.path.append(project_root_directory)


@pytest.mark.parametrize(
    "classification_type, y, y_hat, expected",
    [
        (
            "binary",
            np.array([0.49999, 0.5000001, 0.99999, 0.00001]),
            np.array([0, 1, 0, 1]),
            0.5,
        ),
        (
            "multiclass",
            np.array(
                [
                    [0.6, 0.1, 0.2, 0.1],
                    [0.2, 0.5, 0.15, 0.15],
                    [0.1, 0.1, 0.5, 0.3],
                    [0.4, 0.05, 0.05, 0.5],
                ]
            ),
            np.array([0, 1, 2, 2]),
            0.75,
        ),
        (
            "multilabel",
            np.array([[0.3, 0.6], [0.8, 0.2], [0.7, 0.9]]),
            np.array([[0, 1], [1, 0], [0, 0]]),
            2 / 3,
        ),
        (
            "multilabel",
            np.array([[0.1, 0.99], [0.56, 0.4], [0.75, 0.25]]),
            np.array([[0, 1], [1, 0], [0, 0]]),
            5 / 6,
        ),
    ],
)
def test_accuracy(classification_type, y, y_hat, expected):
    accuracy_func = Accuracy(classification_type)
    assert accuracy_func(y=y, y_hat=y_hat, adjust_y=True) == pytest.approx(
        expected, 0.01
    )


def test_accuracy_fited_binary():
    accuracy_binary = Accuracy("binary")
    y = np.array([0, 1, 1, 0])
    y_hat = np.array([0, 1, 0, 1])
    assert accuracy_binary(y=y, y_hat=y_hat) == pytest.approx(0.5, 0.01)


def test_accuracy_fited_multiclass():
    accuracy_multiclass = Accuracy("multiclass")
    y = np.array([0, 1, 0, 2])
    y_hat = np.array([0, 1, 1, 2])
    assert accuracy_multiclass(y=y, y_hat=y_hat) == pytest.approx(0.75, 0.01)


def test_accuracy_fited_multilabel():
    accuracy_multilabel = Accuracy("multilabel")
    y = np.array([[0, 1], [1, 1], [1, 0]])
    y_hat = np.array([[0, 1], [1, 0], [1, 0]])

    assert accuracy_multilabel(y, y_hat) == pytest.approx(5 / 6, 0.01)


def test_str_representation():
    accuracy_func_default = Accuracy("binary")
    assert str(accuracy_func_default) == "Accuracy"

    accuracy_func_custom = Accuracy("binary", threshold=0.7)
    assert str(accuracy_func_custom) == "Accuracy (threshold = 0.7)"

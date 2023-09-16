import pytest
import numpy as np
from src.Functions.EvaluationFunctions.Classification.classification_evaluation_function import ClassificationEvaluationFunction
from src.Functions.EvaluationFunctions.Classification.accuracy import Accuracy  # Replace with your actual import

@pytest.mark.parametrize("classification_type, y, y_hat, expected", [
    ("binary", np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1]), 0.5),
    ("multiclass", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), np.array([0, 1, 2, 2]), 0.75),
    ("multilabel", np.array([[0, 1], [1, 0], [1, 1]]), np.array([[0, 1], [1, 0], [0, 0]]), 2/3),
    ("multilabel", np.array([[0, 1], [1, 0], [1, 0]]), np.array([[0, 1], [1, 0], [0, 0]]), 2/3)
])

def test_accuracy(classification_type, y, y_hat, expected):
    accuracy_func = Accuracy(classification_type)
    assert accuracy_func(y, y_hat) == pytest.approx(expected, 0.01)

def test_str_representation():
    accuracy_func_default = Accuracy("binary")
    assert str(accuracy_func_default) == "Accuracy"

    accuracy_func_custom = Accuracy("binary", threshold=0.7)
    assert str(accuracy_func_custom) == "Accuracy (threshold = 0.7)"
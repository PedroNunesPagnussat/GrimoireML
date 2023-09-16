import pytest
import numpy as np
from src.Functions.EvaluationFunctions import  ClassificationEvaluationFunction



class MockClassificationEvaluationFunction(ClassificationEvaluationFunction):
    def __init__(self, type: str) -> None:
        super().__init__(type)

    def __call__(self, y: float, y_hat: float) -> float:
        return 0.0

    def __str__(self) -> str:
        return "MockClassificationEvaluationFunction"
    

def test_invalid_type():
    with pytest.raises(ValueError):
        MockClassificationEvaluationFunction("invalid_type")



def test_get_prediction_binary():
    mock_func = MockClassificationEvaluationFunction("binary")
    y_pred = np.array([0.2, 0.7, 0.6, 0.1])
    expected = np.array([0, 1, 1, 0])
    print(mock_func._get_prediction(y_pred))
    assert np.array_equal(mock_func._get_prediction(y_pred), expected)

def test_get_prediction_multiclass():
    mock_func = MockClassificationEvaluationFunction("multiclass")
    y_pred = np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    expected = np.array([2, 0])
    assert np.array_equal(mock_func._get_prediction(y_pred), expected)

def test_get_prediction_multilabel():
    mock_func = MockClassificationEvaluationFunction("multilabel")
    y_pred = np.array([[0.1, 0.7], [0.8, 0.2]])
    expected = np.array([[0, 1], [1, 0]])
    assert np.array_equal(mock_func._get_prediction(y_pred), expected)

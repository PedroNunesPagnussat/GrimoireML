from abc import ABC, abstractmethod
from grimoireml.Functions.function import Function
import numpy as np


class ClassificationEvaluationFunction(Function, ABC):
    """This is the base class for all classification evaluation functions"""

    @abstractmethod
    def __init__(self, type: str, threshold: float = 0.5) -> None:
        types = {
            "binary": 0,
            "multiclass": 1,
            "multilabel": 2
        }
        
        if type not in types:
            raise ValueError("Invalid type")
        
        self._type = types[type]
        self._threshold = threshold
        

    @abstractmethod
    def __call__(self, y: float, y_hat: float) -> float:
        """This method is what calculates the evaluation of a point"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """This method is called when the function is printed"""
        pass


    def _get_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """This method is get the predicted values from the probability vector based on the type of classification"""

        if self._type == 1:
            return np.argmax(y_pred, axis=1)
        
        elif self._type == 0 or self._type == 2:
            return np.where(y_pred >= self._threshold, 1, 0)
        
        





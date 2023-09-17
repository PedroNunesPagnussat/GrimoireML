

from grimoireml.Functions.EvaluationFunctions.Classification.classification_evaluation_function import ClassificationEvaluationFunction
import numpy as np


class Accuracy(ClassificationEvaluationFunction):
    def __init__(self, type: str, threshold: float = 0.5) -> None:
        super().__init__(type, threshold)


    def __call__(self, y: float, y_hat: float, adjust_y: bool = False) -> float:
        """This method is what calculates the evaluation of a point"""
        
        if adjust_y:
            y = super()._get_prediction(y)

        
        if self._type == 0 or self._type == 1:
            return np.mean(y == y_hat)
        
        elif self._type == 2:
            return np.mean(y == y_hat)

        return -1        

    def __str__(self) -> str:
        """This method is called when the function is printed"""
        if self._threshold == 0.5:
            return "Accuracy"
        else:
            return f"Accuracy (threshold = {self._threshold})"
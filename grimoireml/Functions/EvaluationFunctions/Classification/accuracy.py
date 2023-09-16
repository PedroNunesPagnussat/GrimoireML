from grimoireml.Functions.EvaluationFunctions.Classification.classification_evaluation_function import ClassificationEvaluationFunction
import numpy as np


class Accuracy(ClassificationEvaluationFunction):
    def __init__(self, type: str, threshold: float = 0.5) -> None:
        super().__init__(type, threshold)


    def __call__(self, y: float, y_hat: float) -> float:
        """This method is what calculates the evaluation of a point"""
        
        if self._type == 0:
            return np.mean(y == y_hat)
        
        elif self._type == 1:
            return np.mean(np.argmax(y, axis=1) == y_hat)
        
        elif self._type == 2:
            return np.mean(np.all(y == y_hat, axis=1))
        

    def __str__(self) -> str:
        """This method is called when the function is printed"""
        if self._threshold == 0.5:
            return "Accuracy"
        else:
            return f"Accuracy (threshold = {self._threshold})"
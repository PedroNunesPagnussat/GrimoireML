from abc import ABC, abstractmethod
import numpy as np


class EvaluationFunction(ABC):
    """
    Abstract base class for evaluation functions.

    Defines the method signature for evaluation functions, serving as a blueprint for all concrete implementations.
    """
    @abstractmethod
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the evaluation metric based on true and predicted labels.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The calculated evaluation metric.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the string representation of the evaluation function.

        Returns:
            str: The name or a description of the evaluation function.
        """
        pass

    

class MSE(EvaluationFunction):
    """
    Mean Squared Error (MSE) evaluation function for regression problems.
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Mean Squared Error based on true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated Mean Squared Error.
        """


        mse = np.mean((y_true - y_pred) ** 2)

        return mse

    def __str__(self) -> str:
        return "Mean Squared Error (MSE)"

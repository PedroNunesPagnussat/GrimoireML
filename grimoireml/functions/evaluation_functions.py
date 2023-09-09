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


    def _true_positives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """
        Compute the number of true positives.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            int: The number of true positives.
        """
        return np.sum(y_true == y_pred)


class Accuracy(EvaluationFunction):
    """
    Accuracy evaluation function.
    """
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy based on true and predicted labels.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: The calculated accuracy.
        """

        tp = self._true_positives(y_true, y_pred)
        total_predictions = y_true.size

        return tp / total_predictions
    

    def __str__(self) -> str:
        """
        Return the string representation of the evaluation function.

        Returns:
            str: The name or a description of the evaluation function.
        """
        return "Accuracy"
    

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

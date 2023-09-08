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



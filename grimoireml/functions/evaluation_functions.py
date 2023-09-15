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



class MAE(EvaluationFunction):
    """
    Mean Absolute Error (MAE) evaluation function for regression problems.
    """

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Mean Absolute Error based on true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated Mean Absolute Error.

        Note:
            The function assumes that y_true and y_pred have the same shape.
        """

        # Calculating the mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))

        return mae

    def __str__(self) -> str:
        """
        Return the string representation of the evaluation function.

        Returns:
            str: The name or a description of the evaluation function.
        """
        return "Mean Absolute Error (MAE)"
    


class Accuracy(EvaluationFunction):
    """
    Mock Docstring
    """

    def __init__(self, type : str = None, threshold: float = 0.5) -> None:
        """
        Initialize the Accuracy object with optional type and threshold.

        Parameters:
        -----------
        type : str, optional
            The type of prediction (default is None).
        threshold : float, optional
            The threshold for classifying a prediction as positive in binary classification (default is 0.5).
        """
        self._type = type
        self._threshold = threshold

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy based on the type of prediction.

        Parameters:
        -----------
        y_true : np.ndarray
            The ground truth labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The computed accuracy.
        """
        if self._type == "binary":
            return self._compute_binary(y_true, y_pred)
        elif self._type == "multiclass":
            return self._compute_multiclass(y_true, y_pred)
        elif self._type == "multilabel":
            return self._compute_multilabel(y_true, y_pred)
        
        # Further implementation here
        return 0.0

    def _compute_binary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy for binary classification.

        Parameters:
        -----------
        y_true : np.ndarray
            The ground truth labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The computed accuracy for binary classification.
        """
        y_pred = np.where(y_pred >= self._threshold, 1, 0)
        accuracy = np.mean(y_true == y_pred)
        return accuracy

    def _compute_multiclass(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy for multiclass classification.

        Parameters:
        -----------
        y_true : np.ndarray
            The ground truth labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The computed accuracy for multiclass classification.
        """
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true == y_pred_labels)

        return accuracy

    def _compute_multilabel(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the accuracy for multilabel classification.

        Parameters:
        -----------
        y_true : np.ndarray
            The ground truth labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns:
        --------
        float
            The computed accuracy for multilabel classification.
        """
        y_pred = np.where(y_pred >= self._threshold, 1, 0)
        accuracy = np.mean(np.all(y_true == y_pred, axis=1))
        return accuracy
    

    def __str__(self) -> str:
        """
        Return the string representation of the Accuracy object.

        Returns:
        --------
        str
            The string representation.
        """
        return f"Accuracy"
    

class Precision(EvaluationFunction):
    """
    A class for computing the precision of a model's predictions.
    """

    def __init__(self, type: str = None, threshold: float = 0.5) -> None:
        """
        Initialize the Precision object with optional type and threshold.
        """
        self._type = type
        self._threshold = threshold

    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the precision based on the type of prediction.
        """
        if self._type == "binary":
            return self._compute_binary(y_true, y_pred)
        elif self._type == "multiclass":
            return self._compute_multiclass(y_true, y_pred)
        elif self._type == "multilabel":
            return self._compute_multilabel(y_true, y_pred)
        else:
            return 0.0

    def _compute_binary(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the precision for binary classification.
        """
        y_pred = np.where(y_pred >= self._threshold, 1, 0)
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        precision = true_positives / (predicted_positives + 1e-9)  # Adding a small value to avoid division by zero
        return precision
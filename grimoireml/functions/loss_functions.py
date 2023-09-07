from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    
    This class defines the interface that all loss functions must implement.
    It includes methods for calculating the loss and its derivative.
    """
    
    @abstractmethod
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated loss.
        
        Note:
            The shape and data type of the returned array will depend on the specific loss function.
        """
        pass

    @abstractmethod
    def _compute_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the loss function with respect to the predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated derivative of the loss.
        
        Note:
            The shape and data type of the returned array will depend on the specific loss function.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return the string representation of the loss function.
        
        Returns:
            str: The name or description of the loss function.
        """
        pass

class MSE(LossFunction):
    """
    Mean Squared Error (MSE) loss function.
    
    This class implements the Mean Squared Error loss function, commonly used for regression tasks.
    """
    
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the MSE loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated MSE loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _compute_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the MSE loss function.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated derivative of the MSE loss.
        """

        return (2 * (y_pred - y_true))

    def __str__(self) -> str:
        """
        Return the string representation of the MSE loss function.
        
        Returns:
            str: The name "Mean Squared Error".
        """
        return "Mean Squared Error (MSE)"


class MAE(LossFunction):
    """
    Mean Absolute Error (MAE) loss function.
    
    This class implements the Mean Absolute Error loss function, which is commonly used for regression tasks.
    """
    
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the MAE loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated MAE loss.
        """
        return np.mean(np.abs(y_true - y_pred))

    def _compute_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the MAE loss function.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated derivative of the MAE loss.
        """

        return np.sign(y_pred - y_true)

    def __str__(self) -> str:
        """
        Return the string representation of the MAE loss function.
        
        Returns:
            str: The name "Mean Absolute Error".
        """
        return "Mean Absolute Error (MAE)"

class BCE(LossFunction):
    """
    Binary Cross-Entropy (BCE) loss function.
    
    This class implements the Binary Cross-Entropy loss function, which is commonly used for binary classification tasks.
    """
    
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the BCE loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated BCE loss.
        """

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _compute_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the BCE loss function.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated derivative of the BCE loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def __str__(self) -> str:
        """
        Return the string representation of the BCE loss function.
        
        Returns:
            str: The name "Binary Cross-Entropy".
        """
        return "Binary Cross-Entropy (BCE)"


class CCE(LossFunction):
    """
    Categorical Cross-Entropy (CCE) loss function.
    
    This class implements the Categorical Cross-Entropy loss function, which is commonly used for multi-class classification tasks.
    """
    
    def _compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the CCE loss between true and predicted values.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated CCE loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def _compute_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the CCE loss function.
        
        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted target values.
        
        Returns:
            np.ndarray: The calculated derivative of the CCE loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return y_pred - y_true

    def __str__(self) -> str:
        """
        Return the string representation of the CCE loss function.
        
        Returns:
            str: The name "Categorical Cross-Entropy".
        """
        return "Categorical Cross-Entropy (CCE)"

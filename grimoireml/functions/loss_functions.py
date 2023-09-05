from typing import Callable, Tuple
import numpy as np

def get_loss_function(loss: str) -> Tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Return loss function and its derivative based on input.
    
    Args:
        loss (str): The name of the loss function ("MSE", "MAE").
        
    Returns:
        Tuple[Callable, Callable]: The loss function and its derivative.
        
    Raises:
        Exception: If an invalid loss function name is provided.
    """
    if loss not in loss_map:
        raise ValueError(f"Loss function {loss} not supported, supported losses are: {list(loss_map.keys())}")
    return loss_map[loss]

def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Mean Squared Error between true and predicted values.
    
    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        np.ndarray: The Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)

def _mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute derivative of Mean Squared Error.
    
    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        np.ndarray: The derivative of the Mean Squared Error.
    """
    return 2 * (y_pred - y_true)

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Mean Absolute Error between true and predicted values.
    
    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        np.ndarray: The Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def _mae_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute derivative of Mean Absolute Error.
    
    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        np.ndarray: The derivative of the Mean Absolute Error.
    """
    return np.sign(y_pred - y_true)

def _cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute Cross-Entropy Loss between true and predicted values.
    
    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        np.ndarray: The Cross-Entropy Loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



loss_map = {
    "MSE" : (_mse, _mse_derivative),
    "MAE" : (_mae, _mae_derivative)
}
import numpy as np

def get_loss_function(loss):
    """Return loss function and its derivative based on input."""
    if loss == "MSE":
        return _mse, _mse_derivative
    raise Exception("Invalid loss function")

def _mse(y_true, y_pred):
    """Compute Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def _mse_derivative(y_true, y_pred):
    """Compute derivative of Mean Squared Error."""
    return 2 * (y_pred - y_true)

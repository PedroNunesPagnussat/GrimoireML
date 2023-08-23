import numpy as np

def get_loss_function(loss):
    if type(loss) == str:
        if loss == "MSE":
            return mse, mse_derivative
        
    raise Exception("Invalid loss function")

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)
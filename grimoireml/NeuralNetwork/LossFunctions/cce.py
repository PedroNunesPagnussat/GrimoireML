import numpy as np
from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction


class CCE(LossFunction):
    """This is the Categorical Cross Entropy function"""

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y * np.log(y_hat), axis=1, keepdims=True))

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return y_hat - y

    def __str__(self) -> str:
        return "Categorical Cross Entropy"

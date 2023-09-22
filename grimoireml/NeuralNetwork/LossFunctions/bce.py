import numpy as np
from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction


class BCE(LossFunction):
    """This is the Binary Cross Entropy function"""

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return (y_hat - y) / (y_hat * (1 - y_hat))

    def __str__(self) -> str:
        return "Binary Cross Entropy"

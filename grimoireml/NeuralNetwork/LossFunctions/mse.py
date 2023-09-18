import numpy as np
from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction


class MeanSquaredError(LossFunction):
    """This is the Mean Absolute Error loss function"""

    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        from icecream import ic
        return 0.5 * np.sum((y_hat - y) ** 2)
        return np.mean((y_hat - y) ** 2)

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """This method will be called when the object is called"""
        return (2 / y.size) * np.sum((y_hat - y))

    def __str__(self) -> str:
        return "Mean Squared Error"

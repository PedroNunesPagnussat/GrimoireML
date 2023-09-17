from abc import ABC, abstractmethod
from grimoireml.Functions.function import Function


class RegressionEvaluationFunction(Function, ABC):
    """This is the base class for all regression evaluation functions"""

    @abstractmethod
    def __call__(self, y: float, y_hat: float) -> float:
        """This method is what calculates the evaluation of a point"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """This method is called when the function is printed"""
        pass

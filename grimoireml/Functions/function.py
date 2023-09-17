from abc import ABC, abstractmethod
import numpy as np


class Function(ABC):
    """This is the base class for all functions"""

    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        """This method is what calculates the function value at a given point"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """This method is called when the function is printed"""
        pass

from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    
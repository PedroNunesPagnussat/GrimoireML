import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    
    @abstractmethod
    def __init__(self) -> None:
        pass

class Input(Layer):
    pass

class Dense(Layer):
    pass

import numpy as np

class MLP():

    def __init__(self) -> None:
        self.loss = None
        self.loss_derivative = None
        self.optimizer = None
        self.layers = np.array([], dtype=object)


    def add(self, layer):
        self.layers = np.append(self.layers, layer)
    
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

        input_shape = self.layers[0].input_shape
        for i, layer in enumerate(self.layers[1:]):
            layer.initialize_weights(input_shape)
            layer.initialize_biases()
            input_shape = layer.neurons


    def fit(self, X, y, epochs=1, lr=0.01, batch_size=1):
        pass

    def predict(self, X):
        pass

        
    def __str__(self) -> str:
        s = f"MLP with {len(self.layers)} layers:\n"
        s += f"Loss: {self.loss}\n"
        s += f"Optimizer: {self.optimizer}\n"
        for i, layer in enumerate(self.layers):
            s += f"Layer {i}: {layer}\n"

        return s
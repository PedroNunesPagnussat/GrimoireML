import numpy as np
from pedroflow import loss_functions

class MLP():

    def __init__(self) -> None:
        self.loss = None
        self.loss_derivative = None
        self.optimizer = None
        self.layers = np.array([], dtype=object)


    def add(self, layer):
        self.layers = np.append(self.layers, layer)
    
    def compile(self, loss, optimizer):
        self.loss, self.loss_derivative = loss_functions.get_loss_function(loss)
        self.optimizer = optimizer

        input_shape = self.layers[0].input_shape
        for i, layer in enumerate(self.layers[1:]):
            layer.initialize_weights(input_shape)
            layer.initialize_biases()
            input_shape = layer.neurons


    def fit(self, X, y, epochs=1, lr=0.01, batch_size=1):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for i in range(0, len(X), batch_size):

                print(f"Batch {i + 1}/{len(X) // batch_size}")
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # print(batch_X)
                # print(batch_y)

                y_pred = self.forward(batch_X)
                print(y_pred)
                gradients = self.backward(batch_y, y_pred)
                # self.update(lr)

    
    def forward(self, X):
        inputs = X
        for layer in self.layers[1:]:
            z = np.dot(inputs, layer.weights) + layer.biases
            z = np.array(z, dtype=np.float)
            layer.output = np.apply_along_axis(layer.activation, 1, z)  # Apply activation along the correct axis
            layer.output_derivative = np.apply_along_axis(layer.activation_derivative, 1, layer.output)
            inputs = layer.output

        return inputs

    
    def backward(self, y_true, y_pred):
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
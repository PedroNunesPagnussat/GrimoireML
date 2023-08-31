import numpy as np
from ..functions import loss_functions


class Sequential:
    def __init__(self):
        self._loss = None
        self._loss_derivative = None
        self._optimizer = None
        self._layers = np.array([], dtype=object)

    def add(self, layer):
        self._layers = np.append(self._layers, layer)

    def compile(self, loss, optimizer):
        self._loss, self._loss_derivative = loss_functions.get_loss_function(loss)
        self._optimizer = optimizer

        input_shape = self._layers[0]._input_shape
        for layer in self._layers[1:]:
            layer._initialize_weights_and_bias(input_shape)
            input_shape = layer._neurons

    def fit(self, X, y, epochs=1, lr=0.01, batch_size=1):
        for layer in self._layers[1:]:
            layer._delta = np.zeros((batch_size, layer._neurons))

        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                self._layers[0]._output = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                y_pred = self._forward(self._layers[0]._output)
                self._backward(batch_y, y_pred)
                self._update(lr)

    def _forward(self, X):
        inputs = X
        for layer in self._layers[1:]:
            z = np.dot(inputs, layer._weights) + layer._biases
            layer._sum = z
            layer._output = layer._activation(z)  # Assuming vectorized activation function
            inputs = layer._output
        return inputs

    def _backward(self, y_true, y_pred):
        y_true = y_true.reshape(y_pred.shape)
        error = self._loss_derivative(y_true, y_pred)
        output_layer = self._layers[-1]
        output_layer._delta[:] = error * output_layer._activation_derivative(output_layer._sum)
        for i in range(len(self._layers) - 2, 0, -1):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]
            error = np.dot(next_layer._delta, next_layer._weights.T)
            layer._delta[:] = error * layer._activation_derivative(layer._sum)

    def _compute_gradients(self):
        gradients = []
        for i, layer in enumerate(self._layers[1:]):
            prev_layer = self._layers[i]
            weight_gradients = np.dot(prev_layer._output.T, layer._delta)
            bias_gradients = np.sum(layer._delta, axis=0)
            gradients.append((weight_gradients, bias_gradients))
        return gradients

    def _update(self, lr):
        gradients = self._compute_gradients()
        for i, (weight_gradients, bias_gradients) in enumerate(gradients, start=1):
            layer = self._layers[i]
            layer._weights -= lr * weight_gradients
            layer._biases -= lr * bias_gradients

    def predict(self, X):
        inputs = X
        for layer in self._layers[1:]:
            z = np.dot(inputs, layer._weights) + layer._biases
            inputs = layer._activation(z)  # Assuming vectorized activation function
        return inputs

    def __str__(self) -> str:
        s = f"MLP with {len(self._layers)} layers:\n"
        s += f"Loss: {self._loss}\n"
        s += f"Optimizer: {self._optimizer}\n"
        for i, layer in enumerate(self._layers):
            s += f"Layer {i}: {layer}\n"
        return s
from typing import List, Union
import numpy as np
from ..functions import loss_functions

class Sequential:
    def __init__(self):
        """Initialize a Sequential model."""
        self._loss = None
        self._loss_derivative = None
        self._optimizer = None
        self._layers = np.array([], dtype=object)

    def add(self, layer: object) -> None:
        """Add a layer to the model.
        
        Args:
            layer: The layer to be added.
        """
        self._layers = np.append(self._layers, layer)

    def compile(self, loss: str, optimizer: str) -> None:
        """Compile the model by setting the loss function and optimizer.
        
        Args:
            loss: The loss function.
            optimizer: The optimizer.
        """
        self._loss, self._loss_derivative = loss_functions.get_loss_function(loss)
        self._optimizer = optimizer

        input_shape = self._layers[0]._input_shape
        for layer in self._layers[1:]:
            layer._initialize_weights_and_bias(input_shape)
            input_shape = layer._neurons

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, lr: float = 0.01, batch_size: int = 1) -> None:
        """Fit the model to the data.
        
        Args:
            X: The input data.
            y: The labels.
            epochs: The number of epochs.
            lr: The learning rate.
            batch_size: The batch size.
        """
        for layer in self._layers[1:]:
            layer._delta = np.zeros((batch_size, layer._neurons))

        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                self._layers[0]._output = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                y_pred = self._forward(self._layers[0]._output)
                self._backward(batch_y, y_pred)
                self._update(lr)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Perform the forward pass.
        
        Args:
            X: The input data.
        
        Returns:
            The output of the last layer.
        """
        inputs = X
        for layer in self._layers[1:]:
            z = np.dot(inputs, layer._weights) + layer._biases
            layer._sum = z
            layer._output = layer._activation(z)
            inputs = layer._output
        return inputs

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Perform the backward pass.
        
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        """
        y_true = y_true.reshape(y_pred.shape)
        error = self._loss_derivative(y_true, y_pred)
        output_layer = self._layers[-1]
        np.multiply(error, output_layer._activation_derivative(output_layer._sum), out=output_layer._delta)
        for i in range(len(self._layers) - 2, 0, -1):
            layer = self._layers[i]
            next_layer = self._layers[i + 1]
            error = np.dot(next_layer._delta, next_layer._weights.T)
            np.multiply(error, layer._activation_derivative(layer._sum), out=layer._delta)

    def _compute_gradients(self) -> List[tuple]:
        """Compute the gradients for all layers.
        
        Returns:
            A list of tuples containing weight and bias gradients for each layer.
        """
        gradients = []
        for i, layer in enumerate(self._layers[1:]):
            prev_layer = self._layers[i]
            weight_gradients = np.dot(prev_layer._output.T, layer._delta)
            bias_gradients = np.sum(layer._delta, axis=0)
            gradients.append((weight_gradients, bias_gradients))
        return gradients

    def _update(self, lr: float) -> None:
        """Update the weights and biases of all layers.
        
        Args:
            lr: The learning rate.
        """
        gradients = self._compute_gradients()
        for i, (weight_gradients, bias_gradients) in enumerate(gradients, start=1):
            layer = self._layers[i]
            layer._weights -= lr * weight_gradients
            layer._biases -= lr * bias_gradients

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions based on the input data.
        
        Args:
            X: The input data.
        
        Returns:
            The predictions.
        """
        return self._forward(X)

    def __str__(self) -> str:
        """String representation of the model.
        
        Returns:
            A string describing the model.
        """
        s = f"MLP with {len(self._layers)} layers:\n"
        s += f"Loss: {self._loss}\n"
        s += f"Optimizer: {self._optimizer}\n"
        for i, layer in enumerate(self._layers):
            s += f"Layer {i}: {layer}\n"
        return s

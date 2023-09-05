from typing import List, Union
import numpy as np
from ..functions import loss_functions
from .optimizers import Optimizer
from .layers import Layer

class Sequential:
    def __init__(self):
        """
        Initialize an empty Sequential neural network model.
        
        Attributes:
            _loss (callable): Loss function for the model.
            _loss_derivative (callable): Derivative of the loss function.
            _optimizer (Optimizer): Optimizer to adjust model weights.
            _layers (list): List of layers in the model.
        """
        self._loss = None
        self._loss_derivative = None
        self._optimizer = None
        self._layers = []

    def add(self, layer: Layer) -> None:
        """
        Add a layer to the neural network model.
        
        Args:
            layer (Layer): Layer instance to be added to the model.
        """
        self._layers.append(layer)

    def compile(self, loss: str, optimizer: Optimizer) -> None:
        """
        Compile the model by specifying the loss function and optimizer.
        
        Args:
            loss (str): Name of the loss function to use.
            optimizer (Optimizer): Optimizer instance for weight updates.
        """
        self._loss, self._loss_derivative = loss_functions.get_loss_function(loss)
        self._optimizer = optimizer
        input_shape = self._layers[0]._input_shape
        for layer in self._layers[1:]:
            layer._initialize_weights_and_bias(input_shape)
            input_shape = layer._neurons

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 1) -> None:
        """
        Train the model on the given dataset.
        
        Args:
            X (np.ndarray): Input data array.
            y (np.ndarray): Corresponding labels array.
            epochs (int, optional): Number of training epochs. Default is 1.
            batch_size (int, optional): Size of data batches. Default is 1.
        """
        for layer in self._layers[1:]:
            layer._delta = np.zeros((batch_size, layer._neurons))

        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                self._layers[0]._output = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                self._process_batch(X, batch_y)

    def _process_batch(self, batch_X: np.ndarray, batch_y: np.ndarray) -> None:
        """
        Process a single batch of data through forward and backward passes.
        
        Args:
            batch_X (np.ndarray): Input data for the batch.
            batch_y (np.ndarray): Corresponding labels for the batch.
        """
        y_pred = self._forward(batch_X)
        self._backward(batch_y, y_pred)
        self._compute_gradients()
        self._optimizer._update(self._layers)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the network.
        
        Args:
            X (np.ndarray): Input data array.
        
        Returns:
            np.ndarray: Output from the last layer of the network.
        """
        inputs = X
        for layer in self._layers[1:]:
            z = np.dot(inputs, layer._weights) + layer._biases
            layer._sum = z
            layer._output = layer._activation(z)
            inputs = layer._output
        return inputs

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Perform the backward pass to compute gradients.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels from the forward pass.
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
        """
        Compute gradients for all layers in the network.
        
        Returns:
            List[tuple]: List of tuples containing weight and bias gradients for each layer.
        """
        for i, layer in enumerate(self._layers[1:]):
            prev_layer = self._layers[i]
            layer._weights_grad = np.dot(prev_layer._output.T, layer._delta)
            layer._bias_grad = np.sum(layer._delta, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the trained model.
        
        Args:
            X (np.ndarray): Input data array.
        
        Returns:
            np.ndarray: Array of predictions.
        """
        return self._forward(X)

    def __str__(self) -> str:
        """
        Generate a string representation of the model.
        
        Returns:
            str: Description of the model including number of layers, loss function, and optimizer.
        """
        s = f"MLP with {len(self._layers)} layers:\n"
        s += f"Loss: {self._loss}\n"
        s += f"Optimizer: {self._optimizer}\n"
        for i, layer in enumerate(self._layers):
            s += f"Layer {i}: {layer}\n"
        return s

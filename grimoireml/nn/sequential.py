from typing import List, Union
import numpy as np
from ..functions.loss_functions import LossFunction
from .optimizers import Optimizer
from .layers import Layer
from timeit import default_timer as timer



class Sequential:
    def __init__(self):
        """Initialize a Sequential model."""
        self._loss = None
        self._optimizer = None
        self._layers = []

    def add(self, layer: Layer) -> None:
        """Add a layer to the model.
        
        Args:
            layer: The layer to be added.
        """
        self._layers.append(layer)
        # self._layers = np.append(self._layers, layer)


    def compile(self, loss: LossFunction, optimizer: Optimizer) -> None:
        """Compile the model by setting the loss function and optimizer.
        
        Args:
            loss: The loss function.
            optimizer: The optimizer.
        """

        self._loss = loss
        self._optimizer = optimizer


    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 1, verbose: int = 0) -> None:
        """Fit the model to the data.
        
        Args:
            X: The input data.
            y: The labels.
            epochs: The number of epochs.
            lr: The learning rate.
            batch_size: The batch size.
        """
        num_batches = len(X) // batch_size

        for epoch in range(epochs):
            start_time = timer()
            epoch_loss = 0.0
            epoch_acc = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                self._process_batch(batch_X, batch_y)



                print(f"\r{epoch+1}/{epochs} [{i+batch_size}/{len(X)}] - {timer() - start_time:.2f}s - loss: {epoch_loss / (i+1):.4f} - acc: {epoch_acc / (i+1):.4f}", end='')


    def _process_batch(self, batch_X: np.ndarray, batch_y: np.ndarray) -> None:
        """
        Process a single batch of data through forward and backward passes.
        
        Args:
            batch_X (np.ndarray): Input data for the batch.
            batch_y (np.ndarray): Corresponding labels for the batch.
        """
        batch_size = len(batch_X)


        for layer in self._layers:
            layer._delta = np.zeros((batch_size, layer._neurons))
            

        y_pred = self._forward(batch_X)        
        loss = self._loss._compute(y_true=batch_y, y_pred=y_pred)
        loss_deriv = self._loss._compute_derivative(y_true=batch_y, y_pred=y_pred)
        
        self._backward(loss_deriv)
        self._compute_gradients(batch_X)
        self._optimizer._update(self._layers)
        
        return loss



    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        """Perform the forward pass.
        
        Args:
            X: The input data.
        
        Returns:
            The output of the last layer.
        """
        
        for layer in self._layers:
            inputs = layer._forward(inputs)
        return inputs

    def _backward(self, error: float) -> None:
        """
        Perform the backward pass to compute gradients.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels from the forward pass.
        """

        for layer in list(reversed(self._layers)):
            error = layer._backward(error)
        

    def _compute_gradients(self, inputs: np.ndarray) -> List[tuple]:
        """Compute the gradients for all layers.
        
        Returns:
            A list of tuples containing weight and bias gradients for each layer.
        """
        for layer in self._layers:
            layer._compute_gradients(inputs)
            inputs = layer._output


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

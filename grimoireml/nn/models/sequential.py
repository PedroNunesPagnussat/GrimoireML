from typing import List, Tuple
import numpy as np
from ..functions.loss_functions import LossFunction
from ..optimizers import Optimizer
from ..layers import Layer
from ...functions.evaluation_functions import EvaluationFunction
from timeit import default_timer as timer
from .history import History
import pickle



class Sequential:
    def __init__(self):
        """Initialize a Sequential model."""
        self._loss = None
        self._optimizer = None
        self._layers = []
        
        # Summary attributes
        self.history = None
        
        

    def add(self, layer: Layer) -> None:
        """Add a layer to the model.
        
        Args:
            layer: The layer to be added.
        """
        self._layers.append(layer)
        # self._layers = np.append(self._layers, layer)


    def compile(self, optimizer: Optimizer, loss: LossFunction, metrics: List[EvaluationFunction] = None) -> None:
        """Compile the model by setting the loss function and optimizer.
        
        Args:
            loss: The loss function.
            optimizer: The optimizer.
        """

        self._loss = loss
        self._optimizer = optimizer
        

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 1, 
            verbose: int = 1, validation_data: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """Fit the model to the data.
        
        Args:
            X: The input data.
            y: The labels.
            epochs: The number of epochs.
            lr: The learning rate.
            batch_size: The batch size.
            verbose: The verbosity level.
            validation_data: The validation data.
        """

        num_batches = (len(X) // batch_size)
        self.history = History(has_validation=validation_data is not None)

        for epoch in range(epochs):
            start_time = timer()
            epoch_loss = 0.0
            epoch_metric = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                
                batch_loss, batch_metric = self._process_batch(batch_X, batch_y)
                epoch_loss += batch_loss
                epoch_metric += batch_metric

                if verbose:
                    self._print_progress(verbose=verbose, epoch=epoch, epochs=epochs, batch=i, batch_size=batch_size, num_batches=num_batches, start_time=start_time, epoch_loss=epoch_loss, epoch_metric=epoch_metric, data_len = len(X) ,current_batch=i)

            if verbose:
                print()


            self.history.append_epoch(epoch_loss / len(X), epoch_metric / len(X))

            if validation_data:
                val_loss, val_metric = self._evaluate_on_validation_data(validation_data)
                self.history.append_validation(val_loss, val_metric)

                if verbose:
                    print(f" - Val Loss: {val_loss}")


    def _print_progress(self, verbose: int, epoch: int, epochs: int, batch: int, batch_size: int, num_batches: int, start_time: float, epoch_loss: float, epoch_metric: float, current_batch: int, data_len: int):
        """
        Print training progress during each epoch and batch iteration.
        
        Args:
            verbose (int): Verbosity level, 1 for progress bar, 2 for one-line output.
            epoch (int): Current epoch number.
            epochs (int): Total number of epochs.
            batch (int): Index of the first instance in the current batch.
            batch_size (int): Number of instances in a batch.
            num_batches (int): Total number of batches.
            start_time (float): Time when the current epoch started.
            epoch_loss (float): Accumulated loss for the current epoch.
            epoch_metric (float): Accumulated metric for the current epoch.
            current_batch (int): Index of the last instance in the current batch.
            data_len (int): Total number of instances in the data.
        """

        if verbose == 1:
            progress = min(current_batch + batch_size, data_len) / data_len
            num_hashes = int(progress * 25)
            bar = "#" * num_hashes + "-" * (25 - num_hashes)
            s = f"\r Epoch: {epoch+1}/{epochs} Batch: {batch // batch_size}/{num_batches} - Epoch Time: {timer() - start_time:.2f}s - Loss: {epoch_loss / (current_batch+1):.8f} - METRIC NAME: {epoch_metric / (current_batch+1):.8f} - [{bar}]"
            print(f"{s}", end='')
        elif verbose == 2:
            s = f"\r Epoch: {epoch+1}/{epochs} Batch: {batch // batch_size}/{num_batches} - Epoch Time: {timer() - start_time:.2f}s - Loss: {epoch_loss / (current_batch+1):.8f} - METRIC NAME: {epoch_metric / (current_batch+1):.8f}"
            print(s, end='')



    def _evaluate_on_validation_data(self, validation_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.

        Args:
            validation_data (Tuple[np.ndarray, np.ndarray]): A tuple containing the validation features and labels.

        Returns:
            Tuple[float, float]: A tuple containing the validation loss and validation metric.
        """
        val_X, val_y = validation_data
        val_y_pred = self.predict(val_X)  # Assuming _forward is your prediction method
        val_loss = self._loss._compute(y_true=val_y, y_pred=val_y_pred)
        #val_metric = self._metric._compute(y_true=val_y, y_pred=val_y_pred)  # Assuming you have a _metric attribute for evaluation
        val_metric = 0
        return val_loss, val_metric



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
        
        return loss, 0



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
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model on the given data.
        
        Args:
            X: The input data.
            y: The labels.
        
        Returns:
            The loss value.
        """
        y_pred = self.predict(X)
        return self._loss._compute(y_true=y, y_pred=y_pred)
    


    def save_model(self, path: str) -> None:
        """Save the model to a file.
        
        Args:
            path: The path to the file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

        


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

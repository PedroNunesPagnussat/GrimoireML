from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction
from grimoireml.Functions.function import Function
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer
from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Models.ModelUtils.history import History
import numpy as np
from timeit import default_timer as timer

from icecream import ic


class Sequential:
    def __init__(self, layers: list = None):
        self.loss = None
        self.optimizer = None
        self.history = None
        self.layers = []

        if layers is not None:
            self.initialize_layers(layers)

    def initialize_layers(self, layers: list) -> list:
        input_layer = layers[0]
        for layer in layers[1:]:
            input_layer = layer(input_layer)

        self.layers = layers

    def add(self, layer: Layer):
        self.layers.append(layer)
        return layer

    def build(self, loss: LossFunction, optimizer: Optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.layers = np.array(self.layers)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int,
            batch_size: int,
            metrics: list = [],
            validation_data: tuple = None,
            verbose: bool = True,
    ):

        val_data_bool = validation_data is not None
        self.history = History(metrics, val_data_bool)
        n_batches = len(X) // batch_size

        for epoch in range(epochs):
            epoch_start_time = timer()
            epoch_loss = 0
            epoch_metrics = {str(metric): 0 for metric in self.history.metrics_list}

            for batch in range(0, len(X), batch_size):
                batch_X = X[batch: batch + batch_size]
                batch_y = y[batch: batch + batch_size]

                batch_loss, batch_metrics = self.train_on_batch(batch_X, batch_y)
                epoch_loss += batch_loss

                for metric in batch_metrics:
                    s = str(metric)
                    epoch_metrics[s] += batch_metrics[s]

            # for metric in epoch_metrics:
            #     epoch_metrics[metric] /= n_batches

            epoch_time = timer() - epoch_start_time

            epoch_loss /= n_batches

            epoch_end_time = timer()
            epoch_time = epoch_end_time - epoch_start_time  # noqa: F841

            self.history.history["loss"].append(epoch_loss)
            for metric in epoch_metrics:
                self.history.history[metric].append(epoch_metrics[metric] / n_batches)

            val_loss = None
            val_metrics = None
            if val_data_bool:
                val_loss, val_metrics = self.evaluate_on_validation_data(validation_data[0], validation_data[1])
                self.history.history["val_loss"].append(val_loss)
                for metric in val_metrics:
                    self.history.history["val_" + str(metric)].append(val_metrics[metric])

            if verbose:
                self.log_progress(epoch, epoch_loss, epoch_metrics, epoch_time, val_loss, val_metrics)


    def train_on_batch(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.forward_pass(X)
        loss = self.loss(y_hat, y)

        batch_loss_derivative = np.mean(self.loss.derivative(y_hat, y), axis=1, keepdims=True)
        self.backward_pass(batch_loss_derivative)
        self.optimizer.update_params(self.layers)


        metrics = {
            str(metric): metric(y_hat, y, adjust_y=True)
            for metric in self.history.metrics_list
        }
        return loss, metrics

    def predict(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.predict(X)
        return X

    def forward_pass(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward_pass(self, X: np.ndarray):
        for layer in reversed(self.layers):
            X = layer.backward(X)


    def evaluate(self):
        pass

    def evaluate_on_validation_data(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.predict(X)
        loss = self.loss(y_hat, y)

        metrics = {
            str(metric): metric(y_hat, y, adjust_y=True)
            for metric in self.history.metrics_list
        }
        return loss, metrics

    def log_progress(self, epoch_num, epoch_loss, epoch_metrics, epoch_time, val_loss=None, val_metrics=None):
        print("Epoch: ", epoch_num + 1)
        print(f"Epoch Loss: {epoch_loss:.6f}", end=" ")
        for metric in epoch_metrics:
            print(f"Epoch {metric}: {epoch_metrics[metric]:.6f}", end=" ")

        if val_loss is not None:
            print()
            print(f"Validation Loss: {val_loss:.6f}", end=" ")
            for metric in val_metrics:
                print(f"Validation {metric}: {val_metrics[metric]:.6f}", end=" ")

        print()
        print(f"Epoch Time: {epoch_time:.6f}")

    def __str__(self) -> str:
        s = f"Sequential Model with: Layers: {len(self.layers)} \n"
        s += "Optimizer: " + str(self.optimizer) + "\n"
        for layer in self.layers:
            s += str(layer) + "\n"

        return s

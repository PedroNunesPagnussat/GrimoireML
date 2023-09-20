from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction
from grimoireml.Functions.function import Function
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer
from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Models.ModelUtils.history import History
import numpy as np
from timeit import default_timer as timer


class Sequential:
    def __init__(self):
        self.loss = None
        self.optimizer = None
        self.layers = []
        self.history = None

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
            metrics: list = None,
            validation_data: tuple = None,
            verbose: bool = True,
    ):

        n_batches = len(X) // batch_size
        self.history = History(metrics)

        for _ in range(epochs):
            epoch_start_time = timer()
            epoch_loss = 0
            epoch_metrics = {str(metric): 0 for metric in metrics}

            for batch in range(0, n_batches, batch_size):
                batch_X = X[batch: batch + batch_size]
                batch_y = y[batch: batch + batch_size]

                batch_loss, batch_metrics = self.train_on_batch(batch_X, batch_y)
                epoch_loss += batch_loss
                for metric in metrics:
                    epoch_metrics[str(metric)] += batch_metrics[str(metric)]

                # Print Progress

            epoch_end_time = timer()
            epoch_time = epoch_end_time - epoch_start_time
            # epoch time  # noqa: T201
            # self.history.history["loss"].append(epoch_loss)
            # for metric in metrics:
            #     self.history.history[str(metric)].append(epoch_metrics[str(metric)])

    def train_on_batch(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.forward_pass(X)
        self.backward_pass(self.loss.derivative(y_hat, y))
        self.optimizer.update_params(self.layers)

        loss = self.loss(y_hat, y)
        print(y_hat, loss / 2)  # noqa: T201
        return loss, {}

    def predict(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.predict(X)
        return X

    def forward_pass(self, X: np.ndarray):
        for layer in self.layers:
            X = layer._forward(X)
        return X

    def backward_pass(self, X: np.ndarray):
        for layer in reversed(self.layers):
            X = layer._backward(X)

    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def evaluate(self):
        pass

    def evaluate_on_training(self):
        pass

    def save_model(self):
        pass

    def __str__(self) -> str:
        s = f"Sequential Model with: Layers: {len(self.layers)} \n"
        s += "Optimizer: " + str(self.optimizer) + "\n"
        for layer in self.layers:
            s += str(layer) + "\n"

        return s

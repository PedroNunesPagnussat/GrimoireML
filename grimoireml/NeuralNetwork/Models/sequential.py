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
            metrics: list = None,
            validation_data: tuple = None,
            verbose: bool = True,
    ):


        n_batches = len(X) // batch_size
        for metric in metrics:
            metric.ajust_y = True
        self.history = History(metrics, validation_data)


        for epoch in range(epochs):
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

            epoch_time = timer() - epoch_start_time
            self.history.history["loss"].append(epoch_loss / len(X))

            # for metric in metrics:
            #     self.history.history[str(metric)].append(epoch_metrics[str(metric)])

            if verbose:
                self.log_progress(epoch, epoch_loss, epoch_metrics, epoch_time)
            

    def train_on_batch(self, X: np.ndarray, y: np.ndarray):
        y_hat = self.forward_pass(X)
        self.backward_pass(self.loss.derivative(y_hat, y))
        self.optimizer.update_params(self.layers)

        loss = self.loss(y_hat, y)
        metrics = {str(metric): metric(y_hat, y) for metric in self.history.metrics_list}
        return loss, metrics

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


    def log_progress(self, epoch_num, epoch_loss, epoch_metrics, epoch_time):
        print("Epoch: ", epoch_num + 1)
        print(f"Epoch Loss: {epoch_loss:.6f}")
        for metric in epoch_metrics:
            print(f"Epoch {metric}: {epoch_metrics[metric]:.6f}")
        print(f"Epoch Time: {epoch_time:.6f}")

    def __str__(self) -> str:
        s = f"Sequential Model with: Layers: {len(self.layers)} \n"
        s += "Optimizer: " + str(self.optimizer) + "\n"
        for layer in self.layers:
            s += str(layer) + "\n"

        return s

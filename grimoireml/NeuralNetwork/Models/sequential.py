from grimoireml.NeuralNetwork.LossFunctions.loss_function import LossFunction
from grimoireml.Functions.function import Function
from grimoireml.NeuralNetwork.Optimizers.optimizer import Optimizer
from grimoireml.NeuralNetwork.Layers.layer import Layer
from grimoireml.NeuralNetwork.Models.history import History
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

    def compile(self, loss: LossFunction, optimizer: Optimizer):
        self.loss = loss
        self.optimizer = optimizer


    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, metrics:list=None, validation_data: tuple = None, verbose: bool = True):       
        
        n_batches = len(X) // batch_size
        self.history = History(metrics)

        for epoch in range(epochs):
            epoch_start_time = timer()
            epoch_loss = 0
            epoch_metrics = {str(metric): 0 for metric in metrics}

            for i, batch in enumerate(range(n_batches)):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                batch_loss, batch_metrics = self.train_on_batch(batch_X, batch_y)
                epoch_loss += batch_loss
                for metric in metrics:
                    epoch_metrics[str(metric)] += batch_metrics[str(metric)]

                
                # Print Progress

            epoch_end_time = timer()
            # self.history.history["loss"].append(epoch_loss)
            # for metric in metrics:
            #     self.history.history[str(metric)].append(epoch_metrics[str(metric)])
            

    def train_on_batch(self, X: np.ndarray, y: np.ndarray):
        batch_size = len(X)
        
        y_hat = self.foward_pass(X)
        loss = self.loss(y_hat, y)
        self.backward_pass(self.loss.derivative(y_hat, y))
        print(y_hat, loss / 2)
        return 1, 1

    def predict(self, X: np.ndarray):
        for layer in self.layers:
            X = layer.predict(X)
        return X
    
    def foward_pass(self, X: np.ndarray):
        for layer in self.layers:
            X = layer._forward(X)
        return X
    
    def backward_pass(self, X: np.ndarray):
        for layer in reversed(self.layers):
            X = layer._backward(X)

    
    def update_weights(self):
        for layer in self.layers:
            self.optimizer.update(layer)

    def evaluate():
        pass

    def evaluate_on_training():
        pass

    def save_model():
        pass

    def __str__(self) -> str:
        s = f"Sequential Model with: Layes: {len(self.layers)} + \n"
        s += "Optimizer: " + str(self.optimizer) + "\n"
        for layer in self.layers:
            s += str(layer) + "\n"

        return s
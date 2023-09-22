import numpy as np


class History:
    def __init__(self, metrics, validation_data):
        self.metrics_list = metrics
        self.history = {"loss": []}
        for metric in self.metrics_list:
            self.history[metric] = []

        if validation_data is not None:
            self.history["val_loss"] = []
            for metric in self.metrics_list:
                self.history["val_" + metric] = []

    

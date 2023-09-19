import numpy as np


class History:
    def __init__(self, metrics):
        self.metrics_list = metrics
        self.history = {"loss": []}
        for metric in self.metrics_list:
            self.history[metric] = []
